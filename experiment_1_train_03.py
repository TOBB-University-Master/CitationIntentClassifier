import torch
from torch import Generator
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold
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

DATA_PATH = "data/data_v2.csv"
NUMBER_TRIALS = 20
NUMBER_EPOCHS = 50
COMET_PROJECT_NAME_PREFIX = "experiment-1-flat-03"
COMET_WORKSPACE = "ulakbim-cic-train"
COMET_ONLINE_MODE = True
CHECKPOINT_DIR = "checkpoints_v1_03"
DEFAULT_MODEL_INDEX = 0
NUMBER_CPU = 0
K_FOLDS = 5
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
            logging.StreamHandler(sys.stdout)  # <--- Konsola da yaz
        ]
    )


def evaluate(model, data_loader, device, label_names, criterion):
    """
        Modeli değerlendirir ve doğruluk ile sınıflandırma raporu döndürür.
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

            # Kayıp (Loss) hesaplaması
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
        output_dict=True  #
    )

    avg_val_loss = total_val_loss / len(data_loader) if criterion is not None else 0
    val_macro_f1 = report_dict['macro avg']['f1-score']

    return intent_acc, intent_report_str, avg_val_loss, val_macro_f1


def objective(trial):
    """
    Optuna objective fonksiyonu.
    K-Fold Cross-Validation ile hiperparametreleri değerlendirir.
    """

    # --- 1. Hiperparametreleri Ayarla ---
    config = {
        "data_path": DATA_PATH,
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
    experiment.log_parameter("k_folds", K_FOLDS)
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

    full_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=config['data_path'])
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Toplam kayıt sayısı: {len(full_dataset)}, Sınıf sayısı: {num_labels}")

    generator = Generator().manual_seed(config["seed"])

    # Veriyi %80 (Eğitim+Doğrulama) ve %20 (Test) olarak ayır
    # Test seti bu fonksiyonda kullanılmayacak, sadece K-Fold için 'train_val_dataset' kullanılacak.
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, _ = random_split(
        full_dataset, [train_val_size, test_size], generator=generator
    )
    logging.info(f"K-Fold için {len(train_val_dataset)} örnek kullanılacak ({K_FOLDS} kat).")

    # --- 4. K-Fold Çapraz Doğrulama Döngüsü ---
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=config["seed"])
    fold_scores = []  # Her katın en iyi Makro F1 skorunu tutacak liste

    # 'train_val_dataset' üzerinde K-Fold döngüsü
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_dataset)):
        logging.info(f"\n===== KAT (FOLD) {fold + 1}/{K_FOLDS} BAŞLATILIYOR =====")

        # Bu kat için alt veri setlerini (Subset) oluştur
        train_subset = Subset(train_val_dataset, train_indices)
        val_subset = Subset(train_val_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True,
                                  num_workers=config["num_workers"])
        val_loader = DataLoader(val_subset, batch_size=config["batch_size"], num_workers=config["num_workers"])

        # --- 5. Focal Loss için Sınıf Ağırlıklarını Hesapla ---
        # Ağırlıklar her katın *kendi* eğitim verisine göre hesaplanmalı
        try:
            train_labels = [train_val_dataset[i]['label'].item() for i in train_indices]
            unique_labels_in_fold = np.unique(train_labels)

            # Eğer bir kat'ta tüm sınıflar yoksa, tüm sınıfları zorla
            if len(unique_labels_in_fold) < num_labels:
                logging.warning(f"Kat {fold + 1}: Tüm sınıflar bulunmuyor. Ağırlıklar tüm sınıflara göre hesaplanacak.")
                unique_labels_for_calc = np.arange(num_labels)
            else:
                unique_labels_for_calc = unique_labels_in_fold

            class_weights = compute_class_weight('balanced', classes=unique_labels_for_calc, y=train_labels)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            logging.info(f"Kat {fold + 1} için Sınıf Ağırlıkları (Focal Loss): {class_weights}")
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        except Exception as e:
            logging.error(f"Sınıf ağırlıkları hesaplanırken hata: {e}. Standart FocalLoss kullanılacak.")
            criterion = FocalLoss(gamma=2.0)

        # --- 6. Model, Optimizer ve Scheduler'ı Her Kat İçin Sıfırla ---
        model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
        model.transformer.resize_token_embeddings(len(tokenizer))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        # --- LR Warmup ---
        num_training_steps = len(train_loader) * config["epochs"]
        num_warmup_steps = int(num_training_steps * config["warmup_ratio"])

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # --- 7. Eğitim Döngüsü (Early Stopping ile) ---
        best_val_f1_fold = 0.0
        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(config["epochs"]):
            model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Kat {fold + 1} Epoch {epoch + 1}/{config['epochs']}", leave=False)
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
            val_acc, val_report, avg_val_loss, val_macro_f1 = evaluate(
                model, val_loader, device, label_names_list, criterion
            )

            logging.info(
                f"Kat {fold + 1} Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")

            # Comet ML'e bu katın epoch metriklerini logla
            experiment.log_metrics({
                f"fold_{fold + 1}_train_loss": avg_train_loss,
                f"fold_{fold + 1}_val_loss": avg_val_loss,
                f"fold_{fold + 1}_val_macro_f1": val_macro_f1
            }, step=epoch + 1)

            # --- Early Stopping Kontrolü ---
            if val_macro_f1 > best_val_f1_fold:
                best_val_f1_fold = val_macro_f1
                best_epoch = epoch + 1
                epochs_no_improve = 0
                # Bu katın en iyi modelini kaydet
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold_{fold + 1}.pt"))
                logging.info(f"🚀 Kat {fold + 1} - Yeni en iyi Makro F1: {best_val_f1_fold:.4f} (Epoch {epoch + 1})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= PATIENCE:
                logging.info(
                    f"--- Erken Durdurma --- Kat {fold + 1}, {PATIENCE} epoch boyunca iyileşme olmadı. (En iyi Epoch: {best_epoch})")
                break

        # Bu katın en iyi skorunu listeye ekle
        fold_scores.append(best_val_f1_fold)
        experiment.log_metric(f"fold_{fold + 1}_best_macro_f1", best_val_f1_fold)

    # --- 8. Denemenin (Trial) Nihai Sonucunu Hesapla ---
    average_f1 = np.mean(fold_scores)
    logging.info(f"\n--- DENEME #{trial.number} TAMAMLANDI ---")
    logging.info(f"Tüm Katların (Fold) En İyi F1 Skorları: {fold_scores}")
    logging.info(f"Ortalama K-Fold Makro F1: {average_f1:.4f}")

    # Optuna'ya optimize edeceği ortalama skoru döndür
    experiment.log_metric("avg_kfold_macro_f1", average_f1)

    # Tokenizer ve encoder'ı deneme klasörüne kaydet (sadece bir kez)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    with open(os.path.join(output_dir, "trial_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    experiment.end()
    return average_f1


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

    print(f"🚀 Optimizasyon Başlatılıyor (K-Fold CV, Focal Loss, Early Stopping, LR Warmup ile) 🚀")
    print(f"Model: {MODEL_NAME}")
    print(f"Çalışma Adı (Study Name): {study_name}")
    print(f"Veritabanı Dosyası: {storage_path}")
    print("-------------------------------------------------")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        direction="maximize"  # Makro F1'i maksimize ediyoruz
    )

    study.optimize(objective, n_trials=NUMBER_TRIALS)

    print("Optimizasyon tamamlandı.")
    print("En iyi deneme:")
    trial = study.best_trial

    print(f"  Değer (En Yüksek Ortalama K-Fold Makro F1): {trial.value}")
    print("  En İyi Parametreler: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")