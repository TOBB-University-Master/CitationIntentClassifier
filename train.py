import argparse
import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import optuna
from functools import partial
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from comet_ml import Experiment, OfflineExperiment

# Proje modülleri
from config import Config
from dataset import CitationDataset
from generic_model import TransformerClassifier
from FocalLoss import FocalLoss

# Paralel tokenizasyon sorunlarını önlemek için
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ==============================================================================
#                      YARDIMCI FONKSİYONLAR (HELPER)
# ==============================================================================

def setup_logging(log_file):
    """Loglama ayarlarını yapar."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Root logger'ı temizle (tekrarlı logları önlemek için)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_criterion(loss_name, device, class_weights=None):
    """Config'deki seçime göre Loss fonksiyonunu döndürür."""
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    else:
        weights_tensor = None

    if loss_name == "FocalLoss":
        return FocalLoss(alpha=weights_tensor, gamma=2.0)
    elif loss_name == "CrossEntropyLoss_Weighted":
        return nn.CrossEntropyLoss(weight=weights_tensor)
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    else:
        logging.warning(f"Bilinmeyen loss: {loss_name}. Varsayılan FocalLoss kullanılıyor.")
        return FocalLoss(alpha=weights_tensor, gamma=2.0)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, progress_desc):
    """Tek bir epoch eğitimi yapar."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=progress_desc, leave=False)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Token type ids kontrolü (örn: BERT kullanıyorsa)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
        else:
            logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def evaluate_standard(model, loader, device, criterion, label_names):
    """Standart değerlendirme (Accuracy, F1, Loss)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            else:
                logits = model(input_ids, attention_mask)

            if criterion:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0,output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)

    macro_f1 = report_dict['macro avg']['f1-score']
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

    return acc, macro_f1, avg_loss, report_str


# ==============================================================================
#                      EXPERIMENT 1: FLAT CLASSIFICATION
# ==============================================================================

def objective_flat(trial, model_name):
    """Experiment 1 (Düz Sınıflandırma) için Optuna Objective Fonksiyonu."""

    # --- 1. Ayarlar ---
    model_short_name = Config.get_model_short_name(model_name)
    output_dir = f"{Config.CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, "training.log"))

    # Hiperparametreler
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.05, 0.1])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    config_dict = {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_len": Config.MAX_LEN
    }

    # Cihaz
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- 2. Comet ML ---
    if Config.COMET_ONLINE_MODE:
        experiment = Experiment(api_key=Config.COMET_API_KEY,
                                project_name=Config.COMET_PROJECT_NAME,
                                workspace=Config.COMET_WORKSPACE)
    else:
        experiment = OfflineExperiment(project_name=Config.COMET_PROJECT_NAME,
                                       workspace=Config.COMET_WORKSPACE,
                                       log_dir=output_dir)

    experiment.set_name(f"trial_{trial.number}")
    experiment.log_parameters({
        "lr": lr,
        "batch_size": batch_size,
        "model": model_name,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "patience": Config.PATIENCE,
        "seed": Config.SEED,
        "max_len": Config.MAX_LEN
    })

    # --- 3. Veri Yükleme ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})

    logging.info("Veriler yükleniyor (Flat)...")
    train_df = pd.read_csv(Config.DATA_PATH_TRAIN)
    val_df = pd.read_csv(Config.DATA_PATH_VAL)
    test_df = pd.read_csv(Config.DATA_PATH_TEST)

    # Dataset: Task=None (Düz)
    train_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled", data_frame=train_df)
    val_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled", data_frame=val_df)

    num_labels = len(train_dataset.get_label_names())
    label_names = train_dataset.get_label_names()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Config.NUMBER_CPU)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=Config.NUMBER_CPU)

    # --- 4. Model & Loss ---
    model = TransformerClassifier(model_name, num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Class Weights
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    criterion = get_criterion(Config.LOSS_FUNCTION, device, class_weights)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    num_steps = len(train_loader) * Config.NUMBER_EPOCHS
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(num_steps * warmup_ratio),num_training_steps=num_steps)

    # --- 5. Eğitim Döngüsü ---
    best_score = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    for epoch in range(Config.NUMBER_EPOCHS):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device,f"Epoch {epoch + 1}")
        acc, f1, avg_val_loss, val_report = evaluate_standard(model, val_loader, device, criterion, label_names)

        logging.info(f"Epoch {epoch + 1} | Loss: {avg_val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        experiment.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": acc,
            "val_f1": f1
        }, step=epoch + 1)

        # Early Stopping (Config.EVALUATION_METRIC'e göre)
        current_score = acc if Config.EVALUATION_METRIC == "accuracy" else f1

        if current_score > best_score:
            best_score = current_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            # Tokenizer ve Encoder kaydet
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(train_dataset.label_encoder, f)
            logging.info(f"🚀 Yeni en iyi skor: {best_score:.4f}")
            experiment.log_text(f"best_val_report_epoch_{epoch + 1}.txt", val_report)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= Config.PATIENCE:
            logging.info("Erken durdurma tetiklendi.")
            break

    if os.path.exists(best_model_path):
        logging.info(f"En iyi model yükleniyor ve test ediliyor: {best_model_path}")
        test_score = evaluate_flat_test(
            config=config_dict,
            experiment=experiment,
            test_df=test_df,
            device=device,
            model_path=best_model_path,
            encoder_path=encoder_path
        )

        logging.info(f"DENEME {trial.number} SONUCU -> Val Score: {best_score:.4f} | Test Score: {test_score:.4f}")
    else:
        logging.warning("Best model bulunamadı, test adımı atlanıyor.")

    experiment.end()

    return best_score


# ==============================================================================
#                      EXPERIMENT 2 & 3: HIERARCHICAL
# ==============================================================================

def run_hierarchical_stage(task_type, config, trial, train_df, val_df, experiment, device):
    """Binary veya Multiclass aşamasını eğitir."""
    is_binary = (task_type == 'binary')

    # Config'den parametreleri çek
    lr = trial.suggest_float(f"lr_{task_type}", 1e-5, 5e-5, log=True)
    warmup_ratio = trial.suggest_categorical(f"warmup_{task_type}", [0.05, 0.1])
    weight_decay = trial.suggest_float(f"weight_decay_{task_type}", 0.0, 0.1)
    epochs = Config.NUMBER_EPOCHS

    output_dir = os.path.join(config["output_dir_base"], task_type)
    os.makedirs(output_dir, exist_ok=True)

    # Logger ayarla (stage özelinde)
    stage_logger = logging.getLogger(f"{task_type}_logger")
    fh = logging.FileHandler(os.path.join(output_dir, "training.log"), mode='w')
    stage_logger.addHandler(fh)
    stage_logger.setLevel(logging.INFO)

    logging.info(f"--- {task_type.upper()} Eğitimi Başlıyor ---")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})

    # Dataset Oluşturma (Exp 3 için context kontrolü burada yapılır)
    # Eğer Exp 3 ise include_section_in_input=True
    use_context = (Config.EXPERIMENT_ID == 3)

    train_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",data_frame=train_df, task=task_type, include_section_in_input=use_context)
    val_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",data_frame=val_df, task=task_type, include_section_in_input=use_context)

    # Label Encoder Kaydet
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,num_workers=Config.NUMBER_CPU)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=Config.NUMBER_CPU)

    # Model ve Loss
    num_labels = len(train_dataset.get_label_names())
    model = TransformerClassifier(config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    criterion = get_criterion(Config.LOSS_FUNCTION, device, class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_steps = len(train_loader) * epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(num_steps * warmup_ratio),num_training_steps=num_steps)

    # Eğitim Döngüsü
    best_score = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, f"{task_type} Epoch {epoch + 1}")
        acc, f1, val_loss, val_report = evaluate_standard(model, val_loader, device, criterion, train_dataset.get_label_names())

        logging.info(f"[{task_type}] Epoch {epoch + 1} | Loss: {val_loss:.4f} | Acc: {acc:.4f}")
        experiment.log_metrics({
            f"{task_type}_train_loss": train_loss,
            f"{task_type}_val_acc": acc,
            f"{task_type}_val_f1": f1,
            f"{task_type}_val_loss": val_loss
        }, step=epoch + 1)

        current_score = acc if Config.EVALUATION_METRIC == "accuracy" else f1

        if current_score > best_score:
            best_score = current_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)

            experiment.log_text(f"epoch_{epoch + 1}_best_report_{task_type}.txt", val_report)
            experiment.log_metric(f"best_validation_{Config.EVALUATION_METRIC}_{task_type}", best_score, step=epoch + 1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= Config.PATIENCE:
            logging.info(f"[{task_type}] Erken durdurma.")
            break

    return best_score, best_model_path


def evaluate_flat_test(config, experiment, test_df, device, model_path, encoder_path):
    """
    Flat model için Test seti değerlendirmesi, Confusion Matrix ve Hata Analizi.
    """
    logging.info("--- Flat Test Değerlendirmesi Başlıyor ---")

    # 1. Tokenizer ve Model Hazırlığı
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})

    # Encoder Yükle
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    label_names = label_encoder.classes_

    # Modeli Yükle
    model = TransformerClassifier(config["model_name"], num_labels=len(label_names))
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Dataset Oluştur (Exp 3 ise context ekle)
    use_context = (Config.EXPERIMENT_ID == 3)
    test_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                   data_frame=test_df, task=None, include_section_in_input=use_context)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=Config.NUMBER_CPU)

    all_preds, all_labels = [], []
    misclassified_samples = []  # Hatalı örnekleri saklar

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            current_preds = preds.cpu().numpy()
            current_labels = labels.cpu().numpy()

            all_preds.extend(current_preds)
            all_labels.extend(current_labels)

            # --- Hatalı Tahminleri Yakala ---
            # Batch içindeki her bir örnek için kontrol et
            for idx, (p, l) in enumerate(zip(current_preds, current_labels)):
                if p != l:
                    # Orijinal metni geri çöz (decode)
                    raw_text = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                    misclassified_samples.append({
                        "text": raw_text[:500],
                        "true_label": label_names[l],
                        "predicted_label": label_names[p]
                    })

    # Metrikler
    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0,
                                        output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)
    macro_f1 = report_dict['macro avg']['f1-score']

    logging.info(f"🏆 FLAT TEST SKORU - Acc: {acc:.4f} | F1: {macro_f1:.4f}")
    experiment.log_metrics({"test_acc": acc, "test_f1": macro_f1})
    experiment.log_text("test_classification_report.txt", report_str)

    # --- 1. Confusion Matrix Gönder ---
    try:
        experiment.log_confusion_matrix(
            y_true=all_labels,
            y_predicted=all_preds,
            labels=list(label_names),
            title=f"Test Confusion Matrix (Flat)",
            file_name="test_confusion_matrix.json"
        )
    except Exception as e:
        logging.warning(f"Confusion Matrix loglanamadı: {e}")

    # --- 2. Hatalı Tahminleri Tablo Olarak Gönder (CSV) ---
    if misclassified_samples:
        df_errors = pd.DataFrame(misclassified_samples)
        # CSV olarak kaydet ve logla
        error_csv_path = os.path.join(os.path.dirname(model_path), "flat_misclassified_samples.csv")
        df_errors.to_csv(error_csv_path, index=False)
        experiment.log_table(filename="flat_misclassified_samples.csv", tabular_data=df_errors)
        logging.info(f"Hatalı tahmin edilen {len(df_errors)} örnek 'flat_misclassified_samples.csv' olarak yüklendi.")

    return acc if Config.EVALUATION_METRIC == "accuracy" else macro_f1


def evaluate_hierarchical_test(config, experiment, test_df, device, binary_model_path, multiclass_model_path, binary_enc_path, multiclass_enc_path):
    """Test seti üzerinde hiyerarşik değerlendirme."""
    logging.info("--- Hiyerarşik Test Başlıyor ---")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})

    # Encoders Yükle
    with open(binary_enc_path, "rb") as f:
        bin_enc = pickle.load(f)
    with open(multiclass_enc_path, "rb") as f:
        multi_enc = pickle.load(f)

    # Modelleri Yükle
    bin_model = TransformerClassifier(config["model_name"], num_labels=len(bin_enc.classes_))
    bin_model.transformer.resize_token_embeddings(len(tokenizer))
    bin_model.load_state_dict(torch.load(binary_model_path, map_location=device))
    bin_model.to(device);
    bin_model.eval()

    multi_model = TransformerClassifier(config["model_name"], num_labels=len(multi_enc.classes_))
    multi_model.transformer.resize_token_embeddings(len(tokenizer))
    multi_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
    multi_model.to(device);
    multi_model.eval()

    # Test Dataset
    use_context = (Config.EXPERIMENT_ID == 3)
    test_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",data_frame=test_df, task="all", include_section_in_input=use_context)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=Config.NUMBER_CPU)

    # Tahmin Mantığı
    non_bg_id = bin_enc.transform(['non-background'])[0]
    bg_orig_id = test_dataset.label_encoder.transform(['background'])[0]

    all_preds, all_labels = [], []
    misclassified_samples = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # 1. Binary
            bin_logits = bin_model(input_ids, attention_mask)
            bin_preds = torch.argmax(bin_logits, dim=1)

            final_preds = torch.zeros_like(bin_preds)

            # 2. Expert (Non-Background)
            expert_indices = (bin_preds == non_bg_id).nonzero(as_tuple=True)[0]
            if len(expert_indices) > 0:
                exp_in = input_ids[expert_indices]
                exp_att = attention_mask[expert_indices]
                multi_logits = multi_model(exp_in, exp_att)
                multi_preds_raw = torch.argmax(multi_logits, dim=1)

                # ID Mapping
                class_names = multi_enc.inverse_transform(multi_preds_raw.cpu().numpy())
                orig_ids = test_dataset.label_encoder.transform(class_names)
                final_preds[expert_indices] = torch.tensor(orig_ids, device=device)

            # 3. Background
            bg_indices = (bin_preds != non_bg_id).nonzero(as_tuple=True)[0]
            final_preds[bg_indices] = bg_orig_id

            # Batch bazında hatalı tahminleri yakalama döngüsü
            curr_preds_np = final_preds.cpu().numpy()
            curr_labels_np = labels.cpu().numpy()
            label_classes = test_dataset.label_encoder.classes_

            for idx, (p, l) in enumerate(zip(curr_preds_np, curr_labels_np)):
                if p != l:
                    # Metni geri çöz (Token ID -> Text)
                    raw_text = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                    misclassified_samples.append({
                        "text": raw_text[:500],  # Çok uzunsa kırpılabilir
                        "true_label": label_classes[l],
                        "predicted_label": label_classes[p]
                    })

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, output_dict=False, zero_division=0)
    macro_f1 = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)['macro avg']['f1-score']

    logging.info(f"🏆 TEST SKORU - Acc: {acc:.4f} | F1: {macro_f1:.4f}")

    experiment.log_metrics({"test_acc": acc, "test_f1": macro_f1})
    experiment.log_text("combined_hierarchical_test_report.txt", report_str)

    # Hatalı Tahminleri CSV Yapıp Tablo Olarak Yükleme
    if misclassified_samples:
        df_errors = pd.DataFrame(misclassified_samples)

        # Diske kaydetmek isterseniz (Opsiyonel):
        error_csv_path = os.path.join(os.path.dirname(binary_model_path), "../hierarchical_misclassified_samples.csv")
        os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
        df_errors.to_csv(error_csv_path, index=False)

        # Comet'e yükle
        experiment.log_table(filename="hierarchical_misclassified_samples.csv", tabular_data=df_errors)
        logging.info(f"Hatalı tahmin edilen {len(df_errors)} örnek Comet'e yüklendi.")

    try:
        experiment.log_confusion_matrix(
            y_true=all_labels,
            y_predicted=all_preds,
            labels=test_dataset.get_label_names(),
            title=f"Test Confusion Matrix (Exp {Config.EXPERIMENT_ID})",
            file_name="test_confusion_matrix.json"
        )
    except Exception as e:
        logging.warning(f"Confusion Matrix loglanamadı: {e}")

    return acc if Config.EVALUATION_METRIC == "accuracy" else macro_f1


def objective_hierarchical(trial, model_name):
    """Experiment 2 ve 3 için ortak Objective (Hiyerarşik)."""

    model_short_name = Config.get_model_short_name(model_name)
    output_dir_base = f"{Config.CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"
    os.makedirs(output_dir_base, exist_ok=True)
    setup_logging(os.path.join(output_dir_base, "trial.log"))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    # Config Dictionary
    config = {
        "model_name": model_name,
        "output_dir_base": output_dir_base,
        "batch_size": batch_size
    }

    # Comet
    if Config.COMET_ONLINE_MODE:
        experiment = Experiment(api_key=Config.COMET_API_KEY,
                                project_name=Config.COMET_PROJECT_NAME,
                                workspace=Config.COMET_WORKSPACE)
    else:
        experiment = OfflineExperiment(project_name=Config.COMET_PROJECT_NAME,
                                       workspace=Config.COMET_WORKSPACE,
                                       log_dir=output_dir_base)

    experiment.set_name(f"hierarchical_trial_{trial.number}")
    experiment.log_parameters(trial.params)
    experiment.log_parameters({
        "model_name": config["model_name"],
        "patience": Config.PATIENCE,
        "seed": Config.SEED,
        "max_len": Config.MAX_LEN,
        "optimizer": "AdamW"
    })

    # Veri Yükleme
    train_df = pd.read_csv(Config.DATA_PATH_TRAIN)
    val_df = pd.read_csv(Config.DATA_PATH_VAL)
    test_df = pd.read_csv(Config.DATA_PATH_TEST)

    # 1. Binary Model Eğitimi
    best_bin_score, bin_model_path = run_hierarchical_stage("binary", config, trial, train_df, val_df, experiment, device)
    experiment.log_metric("final_binary_best_score", best_bin_score)

    # 2. Multiclass Model Eğitimi
    best_multi_score, multi_model_path = run_hierarchical_stage("multiclass", config, trial, train_df, val_df, experiment, device)
    experiment.log_metric("final_multiclass_best_score", best_multi_score)

    # 3. Birleşik Test
    bin_enc_path = os.path.join(output_dir_base, "binary/label_encoder.pkl")
    multi_enc_path = os.path.join(output_dir_base, "multiclass/label_encoder.pkl")

    final_score = evaluate_hierarchical_test(config, experiment, test_df, device,
                                             bin_model_path, multi_model_path,
                                             bin_enc_path, multi_enc_path)

    # Tokenizer kaydet (trial root)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})
    tokenizer.save_pretrained(output_dir_base)

    experiment.end()
    return final_score


# ==============================================================================
#                      MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--experiment_id", type=int, default=1, help="1: Flat, 2: Hierarchical, 3: Context-Aware")
    parser.add_argument("--model_index", type=int, default=0, help="Index of the model in Config.MODELS")
    args = parser.parse_args()

    # 1. Config Ayarla
    try:
        Config.set_experiment(args.experiment_id)
        Config.set_model(args.model_index)
        Config.print_config()
    except ValueError as e:
        print(f"Hata: {e}")
        sys.exit(1)

    model_name = Config.ACTIVE_MODEL_NAME
    print(f"🚀 Seçilen Model: {model_name}")

    # 2. Study Oluştur
    model_short_name = Config.get_model_short_name(model_name)
    db_path = Config.get_optuna_db_path(model_name)
    storage = f"sqlite:///{db_path}"

    study_name = f"{model_short_name}_exp{args.experiment_id}_study"

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="maximize")

    # 3. Doğru Objective'i Seç
    if args.experiment_id == 1:
        objective_func = partial(objective_flat, model_name=model_name)
    elif args.experiment_id in [2, 3]:
        objective_func = partial(objective_hierarchical, model_name=model_name)
    else:
        print("Geçersiz Experiment ID")
        sys.exit(1)

    # 4. Optimizasyonu Başlat
    print("Optimizasyon Başlıyor...")
    study.optimize(objective_func, n_trials=Config.NUMBER_TRIALS)

    print("✅ En İyi Sonuç:")
    print(f"  Trial: {study.best_trial.number}")
    print(f"  Skor: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()