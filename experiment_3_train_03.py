import argparse
from comet_ml import Experiment
from comet_ml import OfflineExperiment

import torch
import torch.nn as nn
from torch import Generator
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW

import os
import logging
import pickle
import json
import optuna
import sys
import numpy as np

from functools import partial
from transformers import get_scheduler, AutoTokenizer
from collections import Counter
from dataset import CitationDataset
from FocalLoss import FocalLoss
from generic_model import TransformerClassifier
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

"""
    Bu eğitim versiyonunda ek olarak şunlar yapılmaktadır:
    - HEDEF: ACCURACY (Doğruluk)
    - Sabit Train/Val/Test dosyaları
    - Focal Loss kullanımı (sınıf dengesizliği için)
    - Erken Durdurma (Early Stopping)
    - Öğrenme Hızı Isınması (Learning Rate Warmup)
"""

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# -----------------------------------------------------
MODEL_NAMES = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]

COMET_PROJECT_NAME_PREFIX = "experiment-3-rich"
COMET_WORKSPACE = "ulakbim-cic-train-accuracy"
COMET_ONLINE_MODE = True

DATASET_PATH_TRAIN = "data/data_v2_train_ext.csv"
DATASET_PATH_VAL = "data/data_v2_val_ext.csv"
DATASET_PATH_TEST = "data/data_v2_test_ext.csv"

DATA_OUTPUT_PATH = "checkpoints_v3"
DATASET_INFO = False
LOSS_FUNCTION = "CrossEntropyLoss"      # {CrossEntropyLoss, FocalLoss}
NUMBER_TRIALS = 20
NUMBER_EPOCHS = 50
DEFAULT_MODEL_INDEX = 0
NUMBER_CPU = 8
PATIENCE = 10
# -----------------------------------------------------


def setup_logging(log_file):
    """
         Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log)
         hem de konsola yazdırmak için bir loglama sistemi kurar
    """
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
    (Değişiklik yok)
    """
    print(f"\n--- {loader_name} İçin {num_samples} Örnek Veri ---")
    data_iter = iter(data_loader)
    for i in range(num_samples):
        try:
            sample = next(data_iter)
            input_ids = sample['input_ids'][0]
            label = sample['label'][0]
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            print(f"\nÖrnek #{i + 1}:")
            print(f"  Okunabilir Metin: '{decoded_text}'")
            print(f"  Atanmış Label ID: {label.item()}")
        except StopIteration:
            print(f"\nUyarı: '{loader_name}' içinde {num_samples} adetten az veri var.")
            break
    print("-" * (len(loader_name) + 25))


def train_top_level_classifier(config, experiment):
    """
    Modeli 'Background' vs 'Non-Background' olarak ikili sınıflandırma için eğitir.
    En iyi doğrulama F1 skorunu (best_val_f1) döndürür.
    """
    log_file = os.path.join(config["checkpoint_path_binary"], "training_binary.log")
    setup_logging(log_file)
    logging.info("--- ADIM 1: Üst Seviye (İkili) Sınıflandırıcı Eğitimi Başlatılıyor ---")
    logging.info(f"Kullanılan Model: {config['model_name']}")

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

    logging.info("Veri setleri RAM'e ön-yükleniyor...")
    train_data_in_ram = pd.read_csv(config["data_path_train"])
    val_data_in_ram = pd.read_csv(config["data_path_val"])
    logging.info("Veri setleri RAM'e yüklendi.")

    logging.info(f"İkili sınıflandırma için EĞİTİM veri seti yükleniyor: {config['data_path_train']}")
    train_dataset = CitationDataset(
        tokenizer=tokenizer,
        mode="labeled",
        # CSV dosyasından okumak yerine RAM üzerinden alınır
        # csv_path=config['data_path_train'],
        data_frame=train_data_in_ram,
        task='binary',
        include_section_in_input=True
    )

    logging.info(f"İkili sınıflandırma için DOĞRULAMA veri seti yükleniyor: {config['data_path_val']}")
    val_dataset = CitationDataset(
        tokenizer=tokenizer,
        mode="labeled",
        # CSV dosyasından okumak yerine RAM üzerinden alınır
        # csv_path=config['data_path_val'],
        data_frame=val_data_in_ram,
        task='binary',
        include_section_in_input=True
    )

    num_labels = len(train_dataset.get_label_names())
    label_names_list = train_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    with open(config["label_encoder_binary_path"], "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)
    logging.info(f"İkili label encoder şuraya kaydedildi: {config['label_encoder_binary_path']}")

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Binary) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    # --- FOCAL LOSS ve SINIF AĞIRLIKLARI (Binary) ---
    try:
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"İkili Sınıf Ağırlıkları (Focal Loss): {class_weights}")

        if config["loss_function"] == "FocalLoss":
            logging.info("FocalLoss (alpha=weights, gamma=2.0) kullanılıyor.")
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        elif config["loss_function"] == "CrossEntropyLoss":
            logging.info("CrossEntropyLoss (weight=weights) kullanılıyor.")
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            logging.warning(f"Bilinmeyen loss fonksiyonu: {config['loss_function']}. FocalLoss (gamma=2.0) kullanılacak.")
            criterion = FocalLoss(gamma=2.0)

    except Exception as e:
        logging.error(f"Sınıf ağırlıkları hesaplanırken hata: {e}. Standart FocalLoss (alpha=None) kullanılacak.")
        criterion = FocalLoss(gamma=2.0)
    # --- FOCAL LOSS ---

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # --- LR WARMUP ---
    num_training_steps = len(train_loader) * config["epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])

    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)

    # --- EARLY STOPPING Değişkenleri ---
    start_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    logging.info("Yeni bir eğitim başlatılıyor.")
    # --- EARLY STOPPING ---

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} (Binary)")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

        logging.info(f"\nEvaluate başlanıyor...")
        val_acc, val_report, val_macro_f1, avg_val_loss = evaluate(model, val_loader, device, label_names_list,criterion)
        avg_train_loss = total_loss / len(train_loader)

        logging.info(f"\nEpoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Acc): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Macro F1): {val_macro_f1:.4f}\n{val_report}")

        metrics_dict = {
            "binary_train_loss": avg_train_loss,
            "binary_validation_loss": avg_val_loss,
            "binary_validation_accuracy": val_acc,
            "binary_validation_macro_f1": val_macro_f1
        }
        experiment.log_metrics(metrics_dict, step=epoch + 1)

        # --- EARLY STOPPING LOGIC ---
        #if val_macro_f1 > best_val_f1:
        #    best_val_f1 = val_macro_f1
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            logging.info(f"🚀 Yeni en iyi ikili model (Accuracy) kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), config["best_model_path_binary"])
            experiment.log_text(f"epoch_{epoch + 1}_best_report_binary.txt", val_report)
            experiment.log_metric("best_validation_accuracy_binary", best_val_acc, step=epoch + 1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            logging.info(
                f"--- Erken Durdurma (Binary) --- {PATIENCE} epoch boyunca iyileşme olmadı. (En iyi Epoch: {best_epoch})")
            break
        # --- EARLY STOPPING ---

    return best_val_acc


def train_expert_classifier(config, experiment):
    """
    Modeli 'Non-Background' olan 4 sınıf üzerinde eğitir.
    En iyi doğrulama F1 skorunu (best_val_f1) döndürür.
    """
    log_file = os.path.join(config["checkpoint_path_multiclass"], "training_multiclass.log")
    setup_logging(log_file)
    logging.info("\n--- ADIM 2: Uzman (Çok Sınıflı) Sınıflandırıcı Eğitimi Başlatılıyor ---")
    logging.info(f"Kullanılan Model: {config['model_name']}")

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

    logging.info("Veri setleri RAM'e ön-yükleniyor...")
    train_data_in_ram = pd.read_csv(config["data_path_train"])
    val_data_in_ram = pd.read_csv(config["data_path_val"])
    logging.info("Veri setleri RAM'e yüklendi.")

    logging.info(f"Çok sınıflı (Non-Background) EĞİTİM veri seti yükleniyor: {config['data_path_train']}")
    train_dataset = CitationDataset(
        tokenizer=tokenizer,
        mode="labeled",
        # csv_path=config['data_path_train'],
        data_frame=train_data_in_ram,
        task='multiclass',
        include_section_in_input=True
    )

    logging.info(f"Çok sınıflı (Non-Background) DOĞRULAMA veri seti yükleniyor: {config['data_path_val']}")
    val_dataset = CitationDataset(
        tokenizer=tokenizer,
        mode="labeled",
        # csv_path=config['data_path_val'],
        data_frame=val_data_in_ram,
        task='multiclass',
        include_section_in_input=True
    )

    num_labels = len(train_dataset.get_label_names())
    label_names_list = train_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    with open(config["label_encoder_multiclass_path"], "wb") as f:
        pickle.dump(train_dataset.label_encoder, f)
    logging.info(f"Çok sınıflı label encoder şuraya kaydedildi: {config['label_encoder_multiclass_path']}")

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Multiclass) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    # --- FOCAL LOSS ve SINIF AĞIRLIKLARI (Multiclass) ---
    try:
        train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        logging.info(f"Çok Sınıflı Ağırlıklar (Focal Loss): {class_weights}")

        if config["loss_function"] == "FocalLoss":
            logging.info("FocalLoss (alpha=weights, gamma=2.0) kullanılıyor.")
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
        elif config["loss_function"] == "CrossEntropyLoss":
            logging.info("CrossEntropyLoss (weight=weights) kullanılıyor.")
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        else:
            logging.warning(f"Bilinmeyen loss fonksiyonu: {config['loss_function']}. FocalLoss (gamma=2.0) kullanılacak.")
            criterion = FocalLoss(gamma=2.0)
    except Exception as e:
        logging.error(f"Sınıf ağırlıkları hesaplanırken hata: {e}. Standart FocalLoss (alpha=None) kullanılacak.")
        criterion = FocalLoss(gamma=2.0)
    # --- FOCAL LOSS ---

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # --- LR WARMUP ---
    num_training_steps = len(train_loader) * config["epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])

    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)

    # --- EARLY STOPPING Değişkenleri ---
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_epoch = 0
    logging.info("Yeni bir eğitim başlatılıyor.")
    # --- EARLY STOPPING ---

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']} (Multiclass)")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

        logging.info(f"\nEvaluate başlanıyor...")
        val_acc, val_report, val_macro_f1, avg_val_loss = evaluate(model, val_loader, device, label_names_list, criterion)
        avg_train_loss = total_loss / len(train_loader)

        logging.info(f"\nEpoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Acc): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Macro F1): {val_macro_f1:.4f}\n{val_report}")

        metrics_dict = {
            "multiclass_train_loss": avg_train_loss,
            "multiclass_validation_loss": avg_val_loss,
            "multiclass_validation_accuracy": val_acc,
            "multiclass_validation_macro_f1": val_macro_f1
        }
        experiment.log_metrics(metrics_dict, step=epoch + 1)

        # --- EARLY STOPPING LOGIC ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            logging.info(f"🚀 Yeni en iyi uzman model (Accuracy) kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), config["best_model_path_multiclass"])
            experiment.log_text(f"epoch_{epoch + 1}_best_report_multiclass.txt", val_report)
            experiment.log_metric("best_validation_accuracy_multiclass", best_val_acc, step=epoch + 1)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            logging.info(f"--- Erken Durdurma (Multiclass) --- {PATIENCE} epoch boyunca iyileşme olmadı. (En iyi Epoch: {best_epoch})")
            break
        # --- EARLY STOPPING ---

    return best_val_acc


def evaluate_hierarchical(config, experiment):
    """
    Eğitilmiş ikili ve uzman modellerle TEST SETİ üzerinde hiyerarşik birleşik performansı ölçer.
    Karışıklık Matrisini Comet'e log'lar.
    """
    logging.info("\n--- ADIM 3: Birleşik Hiyerarşik TEST Değerlendirmesi Başlatılıyor ---")
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

    logging.info("Veri setleri RAM'e ön-yükleniyor...")
    test_data_in_ram = pd.read_csv(config["data_path_test"])

    # Orijinal (tüm sınıflar) etiket setini yükle
    logging.info(f"Test veri seti yükleniyor: {config['data_path_test']}")
    test_dataset_orig = CitationDataset(
        tokenizer=tokenizer,
        mode="labeled",
        # csv_path=config['data_path_test'],
        data_frame=test_data_in_ram,
        task='all',
        include_section_in_input=True
    )
    orig_label_names = test_dataset_orig.get_label_names()

    # İkili ve Çok Sınıflı görevlerin label encoder'larını YÜKLE
    with open(config["label_encoder_binary_path"], "rb") as f:
        binary_encoder = pickle.load(f)
    with open(config["label_encoder_multiclass_path"], "rb") as f:
        multiclass_encoder = pickle.load(f)

    # İkili modelin "Non-Background" etiketinin ID'sini bul
    non_background_binary_id = list(binary_encoder.transform(['non-background']))[0]

    # Modelleri oluştur ve eğitilmiş en iyi ağırlıkları yükle
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

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Hiyerarşik Test) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")
    test_loader_orig = DataLoader(test_dataset_orig, batch_size=config["batch_size"], num_workers=num_workers)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader_orig:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
                batch["label"].to(device)

            # Adım 1: Üst seviye model ile tahmin yap
            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.zeros_like(binary_preds)

            # Adım 2: Uzman modele danışılacak verileri belirle
            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                expert_input_ids = input_ids[expert_indices]
                expert_attention_mask = attention_mask[expert_indices]

                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)

                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = test_dataset_orig.label_encoder.transform(multiclass_class_names)

                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            # İkili modelin "Background" dediği verilerin etiketini de ekle
            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            background_orig_id = test_dataset_orig.label_encoder.transform(['background'])[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = accuracy_score(all_labels, all_preds)

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

    logging.info(f"🏆 Birleşik Hiyerarşik TEST Başarımı (Accuracy): {overall_accuracy:.4f}")
    logging.info(f"🏆 Birleşik Hiyerarşik TEST Başarımı (Macro F1): {overall_macro_f1:.4f}")
    logging.info(f"Birleşik TEST Sınıflandırma Raporu:\n{report_str}")

    experiment.log_metric("combined_hierarchical_test_accuracy", overall_accuracy)
    experiment.log_metric("combined_hierarchical_test_macro_f1", overall_macro_f1)
    experiment.log_text("combined_hierarchical_test_report.txt", report_str)

    # --- KARIŞIKLIK MATRİSİ (CONFUSION MATRIX) ---
    try:
        logging.info("Karışıklık matrisi oluşturuluyor...")
        experiment.log_confusion_matrix(
            y_true=all_labels,
            y_predicted=all_preds,
            labels=orig_label_names,
            title="Birleşik Karışıklık Matrisi",
            file_name="combined_confusion_matrix.json"
        )

    except Exception as e:
        logging.warning(f"Karışıklık matrisi oluşturulamadı: {e}")

    return overall_accuracy


def objective(trial, model_name):
    """
    Optuna için objective fonksiyonu.
    """

    # --- Hiperparametreler ---
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    warmup_ratio = trial.suggest_categorical("warmup_ratio", [0.05, 0.1])
    # loss_function_name = trial.suggest_categorical("loss_function", ["FocalLoss", "CrossEntropyLoss"])
    loss_function_name = LOSS_FUNCTION
    epochs = NUMBER_EPOCHS

    try:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError):
        num_cpus = NUMBER_CPU

    data_loader_workers = max(0, num_cpus - 1)

    logging.info("=" * 50)
    logging.info("     *** CPU (Worker) KONTROLÜ ***")
    logging.info(f"    os.environ['SLURM_CPUS_PER_TASK']: {os.environ.get('SLURM_CPUS_PER_TASK')}")
    logging.info(f"    NUMBER_CPU (Fallback Değeri): {NUMBER_CPU}")
    logging.info(f"    Kullanılacak 'num_cpus' değeri: {num_cpus}")
    logging.info(f"    Hesaplanan 'data_loader_workers' sayısı: {data_loader_workers}")
    logging.info("=" * 50)

    # --- Config Dosyası ---
    model_short_name = model_name.split('/')[-1]
    output_dir = f"{DATA_OUTPUT_PATH}/{model_short_name}/trial_{trial.number}/"

    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "warmup_ratio": warmup_ratio,
        "loss_function": loss_function_name,
        "model_name": model_name,
        "seed": 42,
        "print_labels": False,
        "num_workers": data_loader_workers,

        "data_path_train": DATASET_PATH_TRAIN,
        "data_path_val": DATASET_PATH_VAL,
        "data_path_test": DATASET_PATH_TEST,

        "checkpoint_path_binary": os.path.join(output_dir, "binary/"),
        "best_model_path_binary": os.path.join(output_dir, "binary/best_model.pt"),
        "label_encoder_binary_path": os.path.join(output_dir, "binary/label_encoder_binary.pkl"),
        "resume_checkpoint_path_binary": os.path.join(output_dir, "binary/training_checkpoint.pt"),

        "checkpoint_path_multiclass": os.path.join(output_dir, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir, "multiclass/best_model.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir, "multiclass/label_encoder_multiclass.pkl"),
        "resume_checkpoint_path_multiclass": os.path.join(output_dir, "multiclass/training_checkpoint.pt")
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
    experiment.log_parameter("num_workers", config["num_workers"])
    experiment.log_parameter("patience", PATIENCE)

    try:
        trial_log_file = os.path.join(output_dir, "trial_summary.log")
        setup_logging(trial_log_file)

        logging.info(f"\n--- DENEME {trial.number} BAŞLATILIYOR ({model_name}) ---")
        logging.info(f"Parametreler: {trial.params}")

        # Adım 1: İkili Modeli (Train/Val) üzerinde eğit
        best_binary_val_acc = train_top_level_classifier(config, experiment)

        # Adım 2: Uzman Modeli (Train/Val) üzerinde eğit
        best_multiclass_val_acc = train_expert_classifier(config, experiment)

        # İki modeli birleştir ve TEST seti üzerinde değerlendir
        overall_test_acc = evaluate_hierarchical(config, experiment)

        setup_logging(trial_log_file)
        logging.info(f"\nDENEME {trial.number} tamamlandı. Ortak yapılandırma dosyaları kaydediliyor...")

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
        tokenizer.add_special_tokens(special_tokens_dict)

        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save_pretrained(output_dir)
        logging.info(f"Tokenizer dosyaları şuraya kaydedildi: {output_dir}")

        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logging.info(f"Yapılandırma dosyası şuraya kaydedildi: {config_path}")

        logging.info(f"DENEME {trial.number} Sonuç: Binary Val Acc: {best_binary_val_acc:.4f}, Multiclass Val Acc: {best_multiclass_val_acc:.4f}")  # <-- GÜNCELLENDİ
        logging.info(f"DENEME {trial.number} Nihai Skor (Test Acc): {overall_test_acc:.4f}")

        experiment.log_metric("final_binary_val_acc", best_binary_val_acc)
        experiment.log_metric("final_multiclass_val_acc", best_multiclass_val_acc)
        experiment.end()

        # Optuna'ya TEST seti üzerindeki skoru döndür
        return overall_test_acc

    except Exception as e:
        try:
            trial_log_file = os.path.join(output_dir, "trial_summary.log")
            setup_logging(trial_log_file)
            logging.error(f"DENEME {trial.number} HATA ALDI: {e}", exc_info=True)
        except Exception as log_e:
            print(f"DENEME {trial.number} KRİTİK HATA: {e}")
            print(f"Loglama hatası: {log_e}")

        if 'experiment' in locals():
            experiment.log_text(f"DENEME_{trial.number}_HATA.txt", str(e))
            experiment.end()

        return 0.0


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
        full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=data_path, task=task)
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


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Classifier Training with Optuna")
    parser.add_argument("--model_index", type=int, default=DEFAULT_MODEL_INDEX, help="Index of the model to train from MODEL_NAMES list.")
    args = parser.parse_args()
    model_index = args.model_index

    try:
        model_name = MODEL_NAMES[model_index]
    except IndexError:
        print(
            f"HATA: Geçersiz model_index: {model_index}. Bu değer 0 ile {len(MODEL_NAMES) - 1} arasında olmalıdır.")
        return

    print(f"\n\n{'=' * 60}")
    print(f"--- BAŞLATILIYOR: {model_name} için {NUMBER_TRIALS} denemelik optimizasyon ---")
    print(f"--- (Focal Loss, Early Stopping, LR Warmup ile) ---")  # <-- YENİ
    print(f"{'=' * 60}\n")

    try:
        model_short_name = model_name.split('/')[-1]
        study_name = f"{model_short_name}_hiearchical_study"

        os.makedirs(DATA_OUTPUT_PATH, exist_ok=True)
        storage_path = f"sqlite:///{os.path.join(DATA_OUTPUT_PATH, f'{model_short_name}_hierarchical.db')}"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="maximize"
        )

        objective_with_model = partial(objective, model_name=model_name)
        study.optimize(objective_with_model, n_trials=NUMBER_TRIALS)

        print(f"\n--- {model_name} İÇİN OPTİMİZASYON TAMAMLANDI ---")
        print(f"En iyi deneme (Best trial): {study.best_trial.number}")
        print(f"En iyi değer (Best value - Test Seti Accuracy): {study.best_value:.4f}")
        print("En iyi parametreler (Best params):")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        model_short_name = model_name.split('/')[-1]
        best_trial_dir = f"{DATA_OUTPUT_PATH}/{model_short_name}/trial_{study.best_trial.number}/"
        print(f"\nEn iyi modelin ve logların kaydedildiği klasör: {best_trial_dir}")

    except Exception as e:
        print(f"KRİTİK HATA: {model_name} için optimizasyon durduruldu. Hata: {e}")


    print(f"\n\n{'=' * 60}")
    print("TÜM MODELLERİN OPTİMİZASYONU TAMAMLANDI.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()