import torch
import os
import logging
import pickle
import json
import optuna
import argparse
import sys
import numpy as np
from functools import partial

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
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

DATA_PATH = "data/data_v2.csv"

CHECKPOINT_DIR = "checkpoints_v2"
COMET_PROJECT_NAME_PREFIX = "experiment-2-hierarchical"
COMET_WORKSPACE = "ulakbim-cic-train"
COMET_ONLINE_MODE = True
DATASET_INFO = False
NUMBER_TRIALS = 20
NUMBER_EPOCHS = 50
DEFAULT_MODEL_INDEX = 0
NUMBER_CPU = 0
PATIENCE = 10
# ==============================================================================


def setup_logging(log_file):
    """
         Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log)
         hem de konsola yazdırmak için bir loglama sistemi kurar
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()  # <-- YENİ: Handler'ları temizle

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)  # <-- YENİ: Konsola da yaz
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

            # <--- YENİ: criterion None olabilir ---
            if criterion is not None:
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=False
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    val_macro_f1 = report_dict['macro avg']['f1-score']
    avg_val_loss = total_val_loss / len(data_loader) if criterion is not None and len(data_loader) > 0 else 0

    return acc, report_str, val_macro_f1, avg_val_loss


def display_samples(loader_name, data_loader, tokenizer, num_samples=1000):
    """
    Verilen bir DataLoader'dan belirtilen sayıda örneği yazdırır.
    (Bu fonksiyonda değişiklik yok)
    """
    print(f"\n--- {loader_name} İçin {num_samples} Örnek Veri ---")
    data_iter = iter(data_loader)
    for i in range(num_samples):
        try:
            sample = next(data_iter)
            input_ids = sample['input_ids'][0]
            label = sample['label'][0]
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"\nÖrnek #{i + 1}:")
            print(f"  Okunabilir Metin: '{decoded_text}'")
            print(f"  Atanmış Label ID: {label.item()}")
        except StopIteration:
            print(f"\nUyarı: '{loader_name}' içinde {num_samples} adetten az veri var.")
            break
    print("-" * (len(loader_name) + 25))


def run_training_stage(config, trial, task_type, experiment):
    """
    Belirtilen görev için (binary veya multiclass) bir eğitim aşamasını çalıştırır.
    (Focal Loss, Early Stopping, LR Warmup eklendi)
    """
    is_binary = task_type == 'binary'

    # Dinamik olarak doğru yolları ve parametreleri seç
    output_dir = config["checkpoint_path_binary"] if is_binary else config["checkpoint_path_multiclass"]
    best_model_path = config["best_model_path_binary"] if is_binary else config["best_model_path_multiclass"]
    lr = config["lr_binary"] if is_binary else config["lr_multiclass"]
    warmup_ratio = config["warmup_ratio_binary"] if is_binary else config["warmup_ratio_multiclass"]
    epochs = config["epochs_binary"] if is_binary else config["epochs_multiclass"]
    encoder_path = config["label_encoder_binary_path"] if is_binary else config["label_encoder_multiclass_path"]

    log_file = os.path.join(output_dir, f"training_{task_type}.log")
    setup_logging(log_file)
    logging.info(f"--- Deneme #{trial.number} - {task_type} Sınıflandırıcı Eğitimi Başlatılıyor ---")

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

    full_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=config['data_path'],
                                   task=task_type)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()

    with open(encoder_path, "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"{task_type.capitalize()} label encoder şuraya kaydedildi: {encoder_path}")

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, _ = random_split(full_dataset, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader ({task_type}) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    # FOCAL LOSS ve SINIF AĞIRLIKLARI
    try:
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        unique_labels = np.unique(train_labels)

        # Sınıf ağırlıklarını 'balanced' modunda hesapla
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"{task_type} için Sınıf Ağırlıkları (Focal Loss): {class_weights}")
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    except Exception as e:
        logging.error(f"Sınıf ağırlıkları hesaplanırken hata: {e}. Standart FocalLoss (alpha=None) kullanılacak.")
        criterion = FocalLoss(gamma=2.0)
    # --- FOCAL LOSS ---

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    # --- LR WARMUP ---
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)

    # --- EARLY STOPPING ---
    start_epoch = 0
    best_val_f1_task = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    # --- EARLY STOPPING ---

    for epoch in range(start_epoch, epochs):
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
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")  # <--- tqdm düzeltmesi

        val_acc, val_report_str, val_macro_f1, avg_val_loss = evaluate(model, val_loader, device, label_names_list, criterion)

        logging.info(f"Epoch {epoch + 1} - {task_type} Train Loss: {total_loss / len(train_loader):.4f}")
        logging.info(f"Epoch {epoch + 1} - {task_type} Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - {task_type} Val Acc (Accuracy): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - {task_type} Val F1 (Macro F1): {val_macro_f1:.4f}")

        avg_train_loss = total_loss / len(train_loader)
        metrics_dict = {
            f"{task_type}_train_loss": avg_train_loss,
            f"{task_type}_validation_loss": avg_val_loss,
            f"{task_type}_validation_accuracy": val_acc,
            f"{task_type}_validation_macro_f1": val_macro_f1
        }
        experiment.log_metrics(metrics_dict, step=epoch + 1)

        # --- EARLY STOPPING LOGIC ---
        if val_macro_f1 > best_val_f1_task:
            best_val_f1_task = val_macro_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            logging.info(f"🚀 Yeni en iyi F1 {task_type} model (Macro F1: {best_val_f1_task:.4f}) kaydediliyor...")
            torch.save(model.state_dict(), best_model_path)

            experiment.log_text(f"epoch_{epoch + 1}_best_report_{task_type}.txt", val_report_str)
            experiment.log_metric(f"best_validation_macro_f1_{task_type}", best_val_f1_task, step=epoch + 1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            logging.info(f"--- Erken Durdurma --- {task_type}, {PATIENCE} epoch boyunca iyileşme olmadı. (En iyi Epoch: {best_epoch})")
            break
        # --- EARLY STOPPING ---

    logging.info(f"--- {task_type} Sınıflandırıcı Eğitimi Tamamlandı ---")


def evaluate_hierarchical(config, experiment):
    """
    Eğitilmiş ikili ve uzman modellerle hiyerarşik birleşik performansı ölçer.
    (Bu fonksiyonda değişiklik yok)
    """
    logging.info("\n--- Birleşik Hiyerarşik Değerlendirme Başlatılıyor ---")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Gerekli tüm bileşenleri yükle
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    full_dataset_orig = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=config['data_path'], task='all')

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

    multiclass_model = TransformerClassifier(model_name=config["model_name"], num_labels=len(multiclass_encoder.classes_))
    multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
    multiclass_model.load_state_dict(torch.load(config["best_model_path_multiclass"], map_location=device))
    multiclass_model.to(device)
    multiclass_model.eval()

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset_orig))
    test_size = len(full_dataset_orig) - train_val_size
    train_val_dataset, _ = random_split(full_dataset_orig, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    _, val_dataset_orig = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Hiyerarşik Değerlendirme) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    val_loader_orig = DataLoader(val_dataset_orig, batch_size=config["batch_size"], num_workers=num_workers)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader_orig:
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

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=orig_label_names,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=orig_label_names,
        zero_division=0,
        output_dict=False
    )

    overall_macro_f1 = report_dict['macro avg']['f1-score']

    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı (Accuracy): {overall_accuracy:.4f}")
    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı (Macro F1): {overall_macro_f1:.4f}")
    logging.info(f"Birleşik Sınıflandırma Raporu:\n{report_str}")

    experiment.log_metric("combined_hierarchical_accuracy", overall_accuracy)
    experiment.log_metric("combined_hierarchical_macro_f1", overall_macro_f1)
    experiment.log_text("combined_hierarchical_report.txt", report_str)

    return overall_macro_f1


def objective(trial, model_name):
    model_short_name = model_name.split('/')[-1]
    output_dir_base = f"{CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"

    try:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError):
        num_cpus = NUMBER_CPU

    data_loader_workers = max(0, num_cpus - 1)

    config = {
        "data_path": DATA_PATH,
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
        "resume_checkpoint_path_binary": os.path.join(output_dir_base, "binary/training_checkpoint.pt"),
        # Not: Bu artık kullanılmıyor
        "label_encoder_binary_path": os.path.join(output_dir_base, "binary/label_encoder.pkl"),

        "checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir_base, "multiclass/best_model.pt"),
        "resume_checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/training_checkpoint.pt"),
        # Not: Bu artık kullanılmıyor
        "label_encoder_multiclass_path": os.path.join(output_dir_base, "multiclass/label_encoder.pkl"),
    }
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

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
    experiment.log_parameter("model_name", config["model_name"])
    experiment.log_parameter("seed", config["seed"])
    experiment.log_parameter("num_workers", data_loader_workers)
    experiment.log_parameter("patience", PATIENCE)  # <--- YENİ

    # 1. Aşama: İkili Modeli Eğit
    run_training_stage(config, trial, 'binary', experiment)

    # 2. Aşama: Uzman Modeli Eğit
    run_training_stage(config, trial, 'multiclass', experiment)

    # 3. Aşama: İki modelin ortak performansını ölç
    overall_macro_f1 = evaluate_hierarchical(config, experiment)

    logging.info(f"Deneme #{trial.number} tamamlandı. Tokenizer ve yapılandırma kaydediliyor...")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.save_pretrained(output_dir_base)

    config_path = os.path.join(output_dir_base, "trial_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    experiment.end()

    return overall_macro_f1


def print_dataset_info(model_name, data_path, seed):
    """
    (Bu fonksiyonda değişiklik yok)
    """
    print("--- Veri Seti Dağılım İncelemesi Başlatılıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def log_class_distribution(subset, name):
        labels = [subset[i]['label'].item() for i in range(len(subset))]
        counts = Counter(labels)
        original_dataset = subset
        while isinstance(original_dataset, Subset):
            original_dataset = original_dataset.dataset
        label_names = original_dataset.label_encoder.classes_
        print(f"\n--- {name} Sınıf Dağılımı ---")
        print(f"Toplam Örnek: {len(subset)}")
        for label_id, count in sorted(counts.items()):
            print(f"    {label_names[label_id]} (ID: {label_id}): {count}")

    for task in ['binary', 'multiclass']:
        print(f"\n{'=' * 20} GÖREV: {task.upper()} {'=' * 20}")
        full_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=data_path, task=task)
        generator = Generator().manual_seed(seed)
        train_val_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=generator)
        train_size = int(0.85 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)
        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Classifier Training with Optuna")
    parser.add_argument("--model_index", type=int, default=DEFAULT_MODEL_INDEX, help="Index of the model to train from MODEL_NAMES list.")
    args = parser.parse_args()
    model_index = args.model_index

    if DATASET_INFO:
        print_dataset_info(model_name=MODEL_NAMES[0], data_path=DATA_PATH, seed=42)
    else:
        try:
            model_name = MODEL_NAMES[model_index]
            print(f"\n\n{'=' * 60}")
            print(f"--- BAŞLATILIYOR: {model_name} için {NUMBER_TRIALS} denemelik optimizasyon ---")
            print(f"--- (Focal Loss, Early Stopping, LR Warmup ile) ---")
            print(f"{'=' * 60}\n")

            model_short_name = model_name.split('/')[-1]
            study_name = f"{model_short_name}_hiearchical_study"
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            storage_path = f"sqlite:///{CHECKPOINT_DIR}/{model_short_name}_hierarchical.db"

            print(f"🚀 Hiyerarşik Optimizasyon Başlatılıyor 🚀")
            print(f"Model: {model_name}")
            print(f"Çalışma Adı (Study Name): {study_name}")
            print(f"Veritabanı Dosyası: {storage_path}")
            print("-------------------------------------------------")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_path,
                load_if_exists=True,
                direction="maximize"
            )

            objective_with_model = partial(objective, model_name=model_name)

            study.optimize(objective_with_model, n_trials=NUMBER_TRIALS)

            print("\nOptimizasyon tamamlandı.")
            print("En iyi deneme:")
            trial = study.best_trial
            print(f"  Değer (En Yüksek Birleşik Macro F1): {trial.value:.4f}")  # <-- Formatlama eklendi
            print("  En İyi Parametreler: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        except IndexError:
            print(
                f"HATA: Geçersiz model_index: {model_index}. Bu değer 0 ile {len(MODEL_NAMES) - 1} arasında olmalıdır.")
            exit(1)
        except Exception as e:
            print(f"KRİTİK HATA: {model_name} için optimizasyon durduruldu. Hata: {e}")

        print(f"\n\n{'=' * 60}")
        print(f"--- {model_name} OPTİMİZASYONU TAMAMLANDI ---")
        print(f"{'=' * 60}")