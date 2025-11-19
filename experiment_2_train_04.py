import torch
import os
import logging
import pickle
import json
import optuna
import argparse
import sys
import numpy as np
import pandas as pd
from functools import partial

from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from torch import nn
from dataset import CitationDataset
from FocalLoss import FocalLoss
from generic_model import TransformerClassifier
from tqdm import tqdm
from comet_ml import Experiment
from comet_ml import OfflineExperiment
from sklearn.utils.class_weight import compute_class_weight

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI ***
# ==============================================================================
MODEL_NAMES = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]

DATA_PATH_TRAIN = "data/data_v2_train_aug.csv"
DATA_PATH_VAL = "data/data_v2_val_aug.csv"
DATA_PATH_TEST = "data/data_v2_test.csv"

CHECKPOINT_DIR = "checkpoints_v2"
COMET_PROJECT_NAME_PREFIX = "experiment-2-hierarchical"
COMET_WORKSPACE = "ulakbim-cic-colab-aug"
COMET_ONLINE_MODE = True

LOSS_FUNCTION = "CrossEntropyLoss_Unweighted"       # Seçenekler: "FocalLoss", "CrossEntropyLoss" , "CrossEntropyLoss_Unweighted"
OPTIMIZATION_METRIC = "accuracy"                    # Seçenekler: "macro_f1", "accuracy"

NUMBER_TRIALS = 20
NUMBER_EPOCHS = 50
DEFAULT_MODEL_INDEX = 0
NUMBER_CPU = 0
PATIENCE = 10
# ==============================================================================


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def evaluate(model, data_loader, device, label_names, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=False
    )

    val_macro_f1 = report_dict['macro avg']['f1-score']
    avg_val_loss = total_val_loss / len(data_loader) if criterion is not None and len(data_loader) > 0 else 0

    return acc, report_str, val_macro_f1, avg_val_loss


def run_training_stage(config, trial, task_type, experiment, train_df, val_df):
    """
    Belirtilen görev için (binary veya multiclass) eğitimi gerçekleştirir.
    Değişiklikler:
    - Veriyi path yerine DataFrame olarak alır.
    - LOSS_FUNCTION değişkenine göre criterion belirler.
    - OPTIMIZATION_METRIC değişkenine göre Early Stopping yapar.
    """
    is_binary = task_type == 'binary'

    output_dir = config["checkpoint_path_binary"] if is_binary else config["checkpoint_path_multiclass"]
    best_model_path = config["best_model_path_binary"] if is_binary else config["best_model_path_multiclass"]
    lr = config["lr_binary"] if is_binary else config["lr_multiclass"]
    warmup_ratio = config["warmup_ratio_binary"] if is_binary else config["warmup_ratio_multiclass"]
    epochs = config["epochs_binary"] if is_binary else config["epochs_multiclass"]
    encoder_path = config["label_encoder_binary_path"] if is_binary else config["label_encoder_multiclass_path"]

    log_file = os.path.join(output_dir, f"training_{task_type}.log")
    setup_logging(log_file)
    logging.info(f"--- Deneme #{trial.number} - {task_type} Sınıflandırıcı Eğitimi ({LOSS_FUNCTION} - Hedef: {OPTIMIZATION_METRIC}) ---")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # --- Veri Seti Oluşturma (DataFrame üzerinden) ---
    train_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled",data_frame=train_df, task=task_type)
    val_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled",data_frame=val_df, task=task_type)

    num_labels = len(train_dataset.get_label_names())
    label_names_list = train_dataset.get_label_names()

    with open(encoder_path, "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    # --- LOSS FUNCTION Seçimi ---
    try:
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        if LOSS_FUNCTION == "CrossEntropyLoss_Weighted":
            logging.info(f"Loss Fonksiyonu: CrossEntropyLoss (AĞIRLIKLI) - Weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        elif LOSS_FUNCTION == "CrossEntropyLoss_Unweighted":
            logging.info(f"Loss Fonksiyonu: CrossEntropyLoss (AĞIRLIKSIZ)")
            criterion = nn.CrossEntropyLoss()

        else:  # Default: FocalLoss
            logging.info(f"Loss Fonksiyonu: FocalLoss (Alpha: {class_weights}, Gamma: 2.0)")
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    except Exception as e:
        logging.error(f"Loss hesaplama hatası: {e}. Fallback -> FocalLoss(gamma=2.0)")
        criterion = FocalLoss(gamma=2.0)

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # --- EARLY STOPPING ---
    best_metric_value = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{epochs} ({task_type})")
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

        val_acc, val_report_str, val_macro_f1, avg_val_loss = evaluate(model, val_loader, device, label_names_list,criterion)

        logging.info(f"Epoch {epoch + 1} [{task_type}] - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_macro_f1:.4f}")

        metrics_dict = {
            f"{task_type}_train_loss": total_loss / len(train_loader),
            f"{task_type}_validation_loss": avg_val_loss,
            f"{task_type}_validation_accuracy": val_acc,
            f"{task_type}_validation_macro_f1": val_macro_f1
        }
        experiment.log_metrics(metrics_dict, step=epoch + 1)

        # --- METRİK SEÇİMİNE GÖRE EARLY STOPPING ---
        current_score = val_acc if OPTIMIZATION_METRIC == "accuracy" else val_macro_f1

        if current_score > best_metric_value:
            best_metric_value = current_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            logging.info(f"🚀 Yeni en iyi {OPTIMIZATION_METRIC} ({current_score:.4f}) kaydediliyor...")
            torch.save(model.state_dict(), best_model_path)
            experiment.log_text(f"epoch_{epoch + 1}_best_report_{task_type}.txt", val_report_str)
            experiment.log_metric(f"best_validation_{OPTIMIZATION_METRIC}_{task_type}", best_metric_value,step=epoch + 1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            logging.info(f"--- Erken Durdurma --- {task_type}, {PATIENCE} epoch iyileşme yok. (En iyi: {best_metric_value:.4f})")
            break

    logging.info(f"--- {task_type} Sınıflandırıcı Eğitimi Tamamlandı ---")


def evaluate_hierarchical(config, experiment, test_df):
    """
    TEST SETİ üzerinde hiyerarşik değerlendirme.
    - OPTIMIZATION_METRIC değerine göre skoru döndürür.
    - Confusion Matrix loglar.
    """
    logging.info("\n--- Birleşik Hiyerarşik Değerlendirme (TEST SETİ) Başlatılıyor ---")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Test verisini DataFrame olarak veriyoruz
    full_dataset_orig = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled",
                                        data_frame=test_df, task='all')

    with open(config["label_encoder_binary_path"], "rb") as f:
        binary_encoder = pickle.load(f)
    with open(config["label_encoder_multiclass_path"], "rb") as f:
        multiclass_encoder = pickle.load(f)

    non_background_binary_id = list(binary_encoder.transform(['non-background']))[0]

    binary_model = TransformerClassifier(model_name=config["model_name"], num_labels=len(binary_encoder.classes_))
    binary_model.transformer.resize_token_embeddings(len(tokenizer))
    binary_model.load_state_dict(torch.load(config["best_model_path_binary"], map_location=device))
    binary_model.to(device)
    binary_model.eval()

    multiclass_model = TransformerClassifier(model_name=config["model_name"],
                                             num_labels=len(multiclass_encoder.classes_))
    multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
    multiclass_model.load_state_dict(torch.load(config["best_model_path_multiclass"], map_location=device))
    multiclass_model.to(device)
    multiclass_model.eval()

    num_workers = config.get("num_workers", 0)
    test_loader = DataLoader(full_dataset_orig, batch_size=config["batch_size"], num_workers=num_workers)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)

            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.zeros_like(binary_preds)
            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                expert_input_ids = input_ids[expert_indices]
                expert_attention_mask = attention_mask[expert_indices]
                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)
                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = full_dataset_orig.label_encoder.transform(multiclass_class_names)
                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            background_orig_id = full_dataset_orig.label_encoder.transform(['background'])[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = accuracy_score(all_labels, all_preds)
    orig_label_names = full_dataset_orig.get_label_names()

    report_dict = classification_report(all_labels, all_preds, target_names=orig_label_names, zero_division=0,output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=orig_label_names, zero_division=0,output_dict=False)
    overall_macro_f1 = report_dict['macro avg']['f1-score']

    logging.info(f"🏆 Birleşik Test Başarımı (Acc): {overall_accuracy:.4f} | (Macro F1): {overall_macro_f1:.4f}")
    logging.info(f"Rapor:\n{report_str}")

    experiment.log_metric("test_accuracy", overall_accuracy)
    experiment.log_metric("test_macro_f1", overall_macro_f1)
    experiment.log_text("test_report.txt", report_str)

    # --- Confusion Matrix ---
    try:
        experiment.log_confusion_matrix(
            y_true=all_labels,
            y_predicted=all_preds,
            labels=orig_label_names,
            title=f"Test Confusion Matrix ({OPTIMIZATION_METRIC})",
            file_name="test_confusion_matrix.json"
        )
    except Exception as e:
        logging.warning(f"Confusion Matrix loglanamadı: {e}")

    # Seçilen metriğe göre dönüş yap
    return overall_accuracy if OPTIMIZATION_METRIC == "accuracy" else overall_macro_f1


def objective(trial, model_name):
    model_short_name = model_name.split('/')[-1]
    output_dir_base = f"{CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"

    try:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", NUMBER_CPU))
    except (KeyError, TypeError):
        num_cpus = NUMBER_CPU
    data_loader_workers = max(0, num_cpus - 1)

    config = {
        "model_name": model_name,
        "seed": 42,
        "num_workers": data_loader_workers,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr_binary": trial.suggest_float("lr_binary", 1e-5, 5e-5, log=True),
        "lr_multiclass": trial.suggest_float("lr_multiclass", 1e-5, 5e-5, log=True),
        "warmup_ratio_binary": trial.suggest_categorical("warmup_ratio_binary", [0.05, 0.1]),
        "warmup_ratio_multiclass": trial.suggest_categorical("warmup_ratio_multiclass", [0.05, 0.1]),
        "epochs_binary": NUMBER_EPOCHS,
        "epochs_multiclass": NUMBER_EPOCHS,
        "checkpoint_path_binary": os.path.join(output_dir_base, "binary/"),
        "best_model_path_binary": os.path.join(output_dir_base, "binary/best_model.pt"),
        "label_encoder_binary_path": os.path.join(output_dir_base, "binary/label_encoder.pkl"),
        "checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir_base, "multiclass/best_model.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir_base, "multiclass/label_encoder.pkl"),
    }
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    # --- Veri Setlerini Hafızaya Yükle (Trial Başına 1 Kez) ---
    logging.info("Veri setleri hafızaya yükleniyor...")
    try:
        train_df = pd.read_csv(DATA_PATH_TRAIN)
        val_df = pd.read_csv(DATA_PATH_VAL)
        test_df = pd.read_csv(DATA_PATH_TEST)
        logging.info(f"Veri yüklendi. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    except Exception as e:
        logging.error(f"Veri okuma hatası: {e}")
        return 0.0

    if COMET_ONLINE_MODE:
        experiment = Experiment(
            api_key="LrkBSXNSdBGwikgVrzE2m73iw",
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{model_short_name}-study",
            workspace=COMET_WORKSPACE,
            auto_log_co2=False,
            auto_output_logging=None
        )
    else:
        experiment = OfflineExperiment(
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{model_short_name}-study",
            workspace=COMET_WORKSPACE,
            auto_log_co2=False,
            auto_output_logging=None
        )

    experiment.set_name(f"trial_{trial.number}")
    experiment.add_tag(model_short_name)
    experiment.log_parameters(trial.params)
    experiment.log_parameter("loss_function", LOSS_FUNCTION)
    experiment.log_parameter("optimization_metric", OPTIMIZATION_METRIC)

    # 1. Aşama: İkili Modeli Eğit
    run_training_stage(config, trial, 'binary', experiment, train_df, val_df)

    # 2. Aşama: Uzman Modeli Eğit
    run_training_stage(config, trial, 'multiclass', experiment, train_df, val_df)

    # 3. Aşama: Test seti üzerinde birleşik performans
    final_score = evaluate_hierarchical(config, experiment, test_df)

    logging.info(f"Deneme #{trial.number} tamamlandı. Skor ({OPTIMIZATION_METRIC}): {final_score:.4f}")

    # Config kaydet
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.save_pretrained(output_dir_base)
    with open(os.path.join(output_dir_base, "trial_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    experiment.end()

    return final_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Classifier Training Refined")
    parser.add_argument("--model_index", type=int, default=DEFAULT_MODEL_INDEX, help="Index of the model")
    args = parser.parse_args()
    model_index = args.model_index

    try:
        model_name = MODEL_NAMES[model_index]
        model_short_name = model_name.split('/')[-1]
        study_name = f"{model_short_name}_refined_study"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        storage_path = f"sqlite:///{CHECKPOINT_DIR}/{model_short_name}_refined.db"

        print(f"🚀 Optimizasyon Başlıyor: {model_name}")
        print(f"Hedef: {OPTIMIZATION_METRIC.upper()} | Loss: {LOSS_FUNCTION}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="maximize"
        )

        objective_with_model = partial(objective, model_name=model_name)
        study.optimize(objective_with_model, n_trials=NUMBER_TRIALS)

        print("En iyi deneme:")
        print(f"  Skor: {study.best_value:.4f}")
        print("  Parametreler: ", study.best_params)

    except Exception as e:
        print(f"Hata: {e}")