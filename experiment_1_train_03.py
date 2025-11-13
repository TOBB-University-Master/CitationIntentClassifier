import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from dataset import CitationDataset
from FocalLoss import FocalLoss
from generic_model import TransformerClassifier
from tqdm import tqdm
from comet_ml import Experiment
from comet_ml import OfflineExperiment

import argparse
import sys
import os
import logging
import json
import pickle
import optuna
import numpy as np

"""
    Bu eğitim versiyonunda ek olarak şunlar yapılmaktadır:
    - Sabit 85/15 Train/Validation bölünmesi KULLANILDI
    - Focal Loss kullanımı (sınıf dengesizliği için)
    - Erken Durdurma (Early Stopping)
    - Öğrenme Hızı Isınması (Learning Rate Warmup)
"""

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI ***
# ==============================================================================
MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]

DATASET_PATH_TRAIN = "data/data_v2_train.csv"
DATASET_PATH_VAL = "data/data_v2_val.csv"

NUMBER_TRIALS = 3
NUMBER_EPOCHS = 5
COMET_PROJECT_NAME_PREFIX = "experiment-1-flat"
#COMET_WORKSPACE = "ulakbim-cic-train"
COMET_WORKSPACE = "ksk-test"
COMET_ONLINE_MODE = True
CHECKPOINT_DIR = "checkpoints_v1_test"
DEFAULT_MODEL_INDEX = 0
NUMBER_CPU = 4
PATIENCE = 10


# ==============================================================================

def setup_logging(log_file):
    """
         Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log)
         hem de konsola yazdırmak için bir loglama sistemi kurar
    """
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
    """
        Modeli değerlendirir ve doğruluk ile sınıflandırma raporu döndürür.
        (Değişiklik yok)
    """
    model.eval()
    all_intent_preds = []
    all_intent_labels = []
    total_val_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)

            intent_logits = model(input_ids, attention_mask)

            if criterion is not None:
                loss = criterion(intent_logits, intent_labels)
                total_val_loss += loss.item()

            intent_preds = torch.argmax(intent_logits, dim=1)

            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())

    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)

    intent_report_str = classification_report(
        all_intent_labels,
        all_intent_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=False
    )

    report_dict = classification_report(
        all_intent_labels,
        all_intent_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    avg_val_loss = total_val_loss / len(data_loader) if criterion is not None and len(data_loader) > 0 else 0
    val_macro_f1 = report_dict['macro avg']['f1-score']

    return intent_acc, intent_report_str, avg_val_loss, val_macro_f1


# ==============================================================================
#                      *** OBJECTIVE (K-FOLD'suz) ***
# ==============================================================================
def objective(trial):
    """
    Optuna objective fonksiyonu.
    Sabit 85/15 Train/Val bölünmesi ile hiperparametreleri değerlendirir.
    """

    # --- 1. Hiperparametreleri Ayarla ---
    config = {
        "data_path_train": DATASET_PATH_TRAIN,
        "data_path_val": DATASET_PATH_VAL,
        "model_name": MODEL_NAME,
        "epochs": NUMBER_EPOCHS,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.05, 0.1]),
        "seed": 42
    }

    try:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError):
        num_cpus = NUMBER_CPU

    data_loader_workers = max(0, num_cpus - 1)
    config["num_workers"] = data_loader_workers

    # --- 2. Çıktı ve Loglama Kurulumu ---
    model_short_name = config["model_name"].split('/')[-1]
    output_dir = f"{CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(output_dir, "trial_summary.log"))

    # --- CPU KONTROL LOGLAMASI ---
    logging.info("=" * 50)
    logging.info("     *** CPU (Worker) KONTROLÜ ***")
    logging.info(f"    os.environ['SLURM_CPUS_PER_TASK']: {os.environ.get('SLURM_CPUS_PER_TASK')}")
    logging.info(f"    NUMBER_CPU (Fallback Değeri): {NUMBER_CPU}")
    logging.info(f"    Kullanılacak 'num_cpus' değeri: {num_cpus}")
    logging.info(f"    Hesaplanan 'data_loader_workers' sayısı: {data_loader_workers}")
    logging.info("=" * 50)
    # --- CPU KONTROL LOGLAMASI ---

    if COMET_ONLINE_MODE:
        experiment = Experiment(
            api_key="LrkBSXNSdBGwikgVrzE2m73iw",
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{model_short_name}-study",
            workspace=COMET_WORKSPACE,
            auto_log_co2=False,
            auto_output_logging=None
        )
    else:
        comet_log_dir = os.path.join(output_dir, "comet_offline_logs")
        os.makedirs(comet_log_dir, exist_ok=True)
        experiment = OfflineExperiment(
            project_name=f"{COMET_PROJECT_NAME_PREFIX}-{model_short_name}-study",
            workspace=COMET_WORKSPACE,
            log_dir=comet_log_dir,
            auto_log_co2=False,
            auto_output_logging=None
        )

    experiment.set_name(f"trial_{trial.number}")
    experiment.add_tag(model_short_name)
    experiment.log_parameters(trial.params)
    experiment.log_parameter("model_name", config["model_name"])
    experiment.log_parameter("seed", config["seed"])
    experiment.log_parameter("num_workers", data_loader_workers)
    experiment.log_parameter("patience", PATIENCE)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config["seed"])
    logging.info(f"--- Deneme #{trial.number} Başlatılıyor ---")
    logging.info(f"Parametreler: {json.dumps(trial.params, indent=4)}")
    logging.info(f"Cihaz seçildi: {device}")

    # --- 3. Veri Setini Yükle ve Böl ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    logging.info(f"Eğitim veri seti yükleniyor: {config['data_path_train']}")
    train_dataset = CitationDataset(
        tokenizer=tokenizer,
        max_len=128,
        mode="labeled",
        csv_path=config['data_path_train']
    )

    logging.info(f"Doğrulama veri seti yükleniyor: {config['data_path_val']}")
    val_dataset = CitationDataset(
        tokenizer=tokenizer,
        max_len=128,
        mode="labeled",
        csv_path=config['data_path_val']
    )

    # Label bilgilerini train_dataset'ten al
    num_labels = len(train_dataset.get_label_names())
    label_names_list = train_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")
    logging.info(f"Eğitim Seti: {len(train_dataset)}, Doğrulama Seti: {len(val_dataset)}")

    # --- 4. DataLoader'ları Oluştur ---
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"])

    # --- 5. Focal Loss için Sınıf Ağırlıklarını Hesapla ---
    try:
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        unique_labels_in_fold = np.unique(train_labels)

        if len(unique_labels_in_fold) < num_labels:
            logging.warning(f"Eğitim setinde tüm sınıflar bulunmuyor. Ağırlıklar tüm sınıflara göre hesaplanacak.")
            unique_labels_for_calc = np.arange(num_labels)
        else:
            unique_labels_for_calc = unique_labels_in_fold

        class_weights = compute_class_weight('balanced', classes=unique_labels_for_calc, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"Sınıf Ağırlıkları (Focal Loss): {class_weights}")
        criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    except Exception as e:
        logging.error(f"Sınıf ağırlıkları hesaplanırken hata: {e}. Standart FocalLoss kullanılacak.")
        criterion = FocalLoss(gamma=2.0)

    # --- 6. Model, Optimizer ve Scheduler'ı Kur ---
    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    num_training_steps = len(train_loader) * config["epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- 7. Eğitim Döngüsü ---
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)

            intent_logits = model(input_ids, attention_mask)
            loss = criterion(intent_logits, intent_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Değerlendirme
        val_acc, val_report, avg_val_loss, val_macro_f1 = evaluate( model, val_loader, device, label_names_list, criterion )

        logging.info(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")

        # Comet ML'e metrikleri logla (fold_ prefix'leri kaldırıldı)
        experiment.log_metrics({
            "train_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "validation_macro_f1": val_macro_f1,
            "validation_accuracy": val_acc
        }, step=epoch + 1)

        # --- Early Stopping Kontrolü ---
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), os.path.join(output_dir, f"best_model.pt"))
            logging.info(f"🚀 Yeni en iyi Makro F1: {best_val_f1:.4f} (Epoch {epoch + 1})")
            experiment.log_metric("best_validation_macro_f1", best_val_f1, step=epoch + 1)
            experiment.log_text(f"epoch_{epoch + 1}_best_report.txt", val_report)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            logging.info(f"--- Erken Durdurma --- {PATIENCE} epoch boyunca iyileşme olmadı. (En iyi Epoch: {best_epoch})")
            break

    # --- 8. Deneme (Trial) Sonucu ---
    logging.info(f"\n--- DENEME #{trial.number} TAMAMLANDI ---")
    logging.info(f"En iyi Makro F1: {best_val_f1:.4f}")

    # Tokenizer ve encoder'ı deneme klasörüne kaydet
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)
    with open(os.path.join(output_dir, "trial_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    experiment.end()
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Modeli Eğitimi için Hiperparametre Optimizasyonu")
    parser.add_argument('--model_index',
                        type=int,
                        default=DEFAULT_MODEL_INDEX,
                        help=f'Eğitilecek modelin MODELS listesindeki indeksi (0-{len(MODELS) - 1} arası).')
    args = parser.parse_args()
    model_index = args.model_index

    if not 0 <= model_index < len(MODELS):
        print(f"HATA: Geçersiz model indeksi: {model_index}. İndeks 0 ile {len(MODELS) - 1} arasında olmalıdır.")
        sys.exit(1)

    MODEL_NAME = MODELS[model_index]
    absolute_checkpoint_dir = os.path.abspath(CHECKPOINT_DIR)
    model_short_name = MODEL_NAME.split('/')[-1]
    study_name = f"{model_short_name}_study"

    os.makedirs(absolute_checkpoint_dir, exist_ok=True)
    storage_path = f"sqlite:///{absolute_checkpoint_dir}/{model_short_name}.db"

    print(f"🚀 Optimizasyon Başlatılıyor (Focal Loss, Early Stopping, LR Warmup ile - K-FOLD YOK) 🚀")
    print(f"Model: {MODEL_NAME}")
    print(f"Çalışma Adı (Study Name): {study_name}")
    print(f"Veritabanı Dosyası: {storage_path}")
    print("-------------------------------------------------")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize"
    )

    study.optimize(objective, n_trials=NUMBER_TRIALS)

    print("Optimizasyon tamamlandı.")
    print("En iyi deneme:")
    trial = study.best_trial

    print(f"  Değer (En Yüksek Validation Makro F1): {trial.value}")  # <-- Güncellendi
    print("  En İyi Parametreler: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")