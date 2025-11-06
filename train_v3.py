import argparse
import torch
import torch.nn as nn
import os
import logging
import pickle
import json
import optuna  # Hyperparameter optimizasyonu için
from functools import partial

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
from dataset import CitationDataset
from generic_model import TransformerClassifier
from tqdm import tqdm

# -----------------------------------------------------
MODEL_NAMES = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]
DATA_PATH = "data/data_v3.csv"
DATA_OUTPUT_PATH = "checkpoints_v3_trials/"
DATASET_INFO = False
NUMBER_TRIALS = 20
NUMBER_EPOCHS = 50
DEFAULT_MODEL_INDEX = 4
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
            logging.StreamHandler()
        ]
    )


def evaluate(model, data_loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0)

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

    return acc, report_str, val_macro_f1


def display_samples(loader_name, data_loader, tokenizer, num_samples=1000):
    """
    Verilen bir DataLoader'dan belirtilen sayıda örneği yazdırır.
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


# TODO: Üst Seviye Sınıflandırıcı Eğitimi
def train_top_level_classifier(config):
    """
    Modeli 'Background' vs 'Non-Background' olarak ikili sınıflandırma için eğitir.
    En iyi doğrulama başarımını (best_val_acc) döndürür.
    """
    log_file = os.path.join(config["checkpoint_path_binary"], "training_binary.log")
    setup_logging(log_file)
    logging.info("--- ADIM 1: Üst Seviye (İkili) Sınıflandırıcı Eğitimi Başlatılıyor ---")
    logging.info(f"Kullanılan Model: {config['model_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    logging.info("İkili sınıflandırma için veri seti yükleniyor...")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=DATA_PATH, task='binary',
                                   include_section_in_input=True)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    with open(config["label_encoder_binary_path"], "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"İkili label encoder şuraya kaydedildi: {config['label_encoder_binary_path']}")

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )

    logging.info("Eğitim/Doğrulama seti, %85 Eğitim ve %15 Doğrulama olarak ayrılıyor...")
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Binary) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    if config["print_labels"]:
        label_names = full_dataset.label_encoder.classes_
        display_samples("Eğitim Seti (Binary)", train_loader, tokenizer, num_samples=2)
        display_samples("Doğrulama Seti (Binary)", val_loader, tokenizer, num_samples=2)
        display_samples("Test Seti (Binary)", test_loader, tokenizer, num_samples=2)

        def log_class_distribution(subset, name):
            labels = [subset[i]['label'].item() for i in range(len(subset))]
            counts = Counter(labels)
            label_names = subset.dataset.label_encoder.classes_ if not isinstance(subset.dataset,
                                                                                  Subset) else subset.dataset.dataset.label_encoder.classes_
            logging.info(f"--- {name} Sınıf Dağilımı ---")
            logging.info(f"Toplam Örnek: {len(subset)}")
            for label_id, count in sorted(counts.items()):
                original_dataset = subset
                while isinstance(original_dataset, Subset):
                    original_dataset = original_dataset.dataset
                label_names = original_dataset.label_encoder.classes_
                logging.info(f"    {label_names[label_id]} (ID: {label_id}): {count}")

        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")
        exit(0)

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * config["epochs"]
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    start_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    if os.path.exists(config["resume_checkpoint_path_binary"]):
        checkpoint = torch.load(config["resume_checkpoint_path_binary"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        logging.info(f"Checkpoint bulundu, eğitime {start_epoch}. epoch'tan devam ediliyor.")
    else:
        logging.info("Yeni bir eğitim başlatılıyor, checkpoint bulunamadı.")

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

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1
        }
        torch.save(checkpoint_data, config["resume_checkpoint_path_binary"])

        val_acc, val_report, val_macro_f1 = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"\nEpoch {epoch + 1} - Doğrulama Başarımı (Acc): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Macro F1): {val_macro_f1:.4f}\n{val_report}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            logging.info(f"🚀 Yeni en iyi ikili model (Macro F1) kaydediliyor: {best_val_f1:.4f}")
            torch.save(model.state_dict(), config["best_model_path_binary"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi uzman (Accuracy): {best_val_acc:.4f} (Not: Model F1'e göre kaydedilir)")

    logging.info("\n--- İkili Model Test Süreci ---")
    model.load_state_dict(torch.load(config['best_model_path_binary']))
    test_acc, test_report, test_macro_f1 = evaluate(model, test_loader, device, label_names_list)
    logging.info(f"\n--- İKİLİ TEST SONUÇLARI ---\nTest Başarımı (Acc): {test_acc:.4f}\nTest Başarımı (Macro F1): {test_macro_f1:.4f}\n{test_report}")

    return best_val_f1


# TODO: Uzman Sınıflandırıcı Eğitimi
def train_expert_classifier(config):
    """
    Modeli 'Non-Background' olan 4 sınıf üzerinde eğitir.
    En iyi doğrulama başarımını (best_val_acc) döndürür.
    """
    log_file = os.path.join(config["checkpoint_path_multiclass"], "training_multiclass.log")
    setup_logging(log_file)
    logging.info("\n--- ADIM 2: Uzman (Çok Sınıflı) Sınıflandırıcı Eğitimi Başlatılıyor ---")
    logging.info(f"Kullanılan Model: {config['model_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    logging.info("Çok sınıflı (Non-Background) veri seti yükleniyor...")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=DATA_PATH, task='multiclass',
                                   include_section_in_input=True)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    with open(config["label_encoder_multiclass_path"], "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"Çok sınıflı label encoder şuraya kaydedildi: {config['label_encoder_multiclass_path']}")

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )

    logging.info("Eğitim/Doğrulama seti, %85 Eğitim ve %15 Doğrulama olarak ayrılıyor...")
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    num_workers = config.get("num_workers", 0)
    logging.info(f"DataLoader (Multiclass) için {num_workers} adet worker (CPU çekirdeği) kullanılacak.")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], num_workers=num_workers)

    if config["print_labels"]:
        label_names = full_dataset.label_encoder.classes_
        display_samples("Eğitim Seti (Binary)", train_loader, tokenizer, num_samples=2)
        display_samples("Doğrulama Seti (Binary)", val_loader, tokenizer, num_samples=2)
        display_samples("Test Seti (Binary)", test_loader, tokenizer, num_samples=2)

        def log_class_distribution(subset, name):
            labels = [subset[i]['label'].item() for i in range(len(subset))]
            counts = Counter(labels)
            label_names = subset.dataset.label_encoder.classes_ if not isinstance(subset.dataset,
                                                                                  Subset) else subset.dataset.dataset.label_encoder.classes_
            logging.info(f"--- {name} Sınıf Dağilımı ---")
            logging.info(f"Toplam Örnek: {len(subset)}")
            for label_id, count in sorted(counts.items()):
                original_dataset = subset
                while isinstance(original_dataset, Subset):
                    original_dataset = original_dataset.dataset
                label_names = original_dataset.label_encoder.classes_
                logging.info(f"    {label_names[label_id]} (ID: {label_id}): {count}")

        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")
        exit(0)

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * config["epochs"]
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    start_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    if os.path.exists(config["resume_checkpoint_path_multiclass"]):
        checkpoint = torch.load(config["resume_checkpoint_path_multiclass"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        logging.info(f"Checkpoint bulundu, eğitime {start_epoch}. epoch'tan devam ediliyor.")
    else:
        logging.info("Yeni bir eğitim başlatılıyor, checkpoint bulunamadı.")

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

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1
        }
        torch.save(checkpoint_data, config["resume_checkpoint_path_multiclass"])

        val_acc, val_report, val_macro_f1 = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"\nEpoch {epoch + 1} - Doğrulama Başarımı (Acc): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - Doğrulama Başarımı (Macro F1): {val_macro_f1:.4f}\n{val_report}")

        # Koşulu F1'e göre güncelle
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            logging.info(f"🚀 Yeni en iyi uzman model (Macro F1) kaydediliyor: {best_val_f1:.4f}")
            torch.save(model.state_dict(), config["best_model_path_multiclass"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"⭐ Yeni en iyi uzman (Accuracy): {best_val_acc:.4f} (Not: Model F1'e göre kaydedilir)")

    logging.info("\n--- Uzman Model Test Süreci ---")
    model.load_state_dict(torch.load(config['best_model_path_multiclass']))
    test_acc, test_report, test_macro_f1 = evaluate(model, test_loader, device, label_names_list)
    logging.info(f"\n--- UZMAN TEST SONUÇLARI ---\nTest Başarımı (Acc): {test_acc:.4f}\nTest Başarımı (Macro F1): {test_macro_f1:.4f}\n{test_report}")

    return best_val_f1


def objective(trial, model_name):
    """
    Optuna için objective fonksiyonu.
    Her deneme için hiperparametreleri belirler, eğitimi çalıştırır
    ve optimize edilecek skoru (uzman modelin val acc) döndürür.
    """

    # --- Hiperparametreleri Belirle ---
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    epochs = NUMBER_EPOCHS

    try:
        num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError):
        num_cpus = 4

    data_loader_workers = max(0, num_cpus - 1)

    # --- Config Dosyasını Oluştur ---
    model_short_name = model_name.split('/')[-1]
    output_dir = os.path.join(DATA_OUTPUT_PATH, f"trial_{trial.number}_{model_short_name}/")

    config = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "model_name": model_name,
        "seed": 42,
        "print_labels": False,
        "num_workers": data_loader_workers,

        # Adım 1 için dinamik yollar
        "checkpoint_path_binary": os.path.join(output_dir, "binary/"),
        "best_model_path_binary": os.path.join(output_dir, "binary/best_model.pt"),
        "label_encoder_binary_path": os.path.join(output_dir, "binary/label_encoder_binary.pkl"),
        "resume_checkpoint_path_binary": os.path.join(output_dir, "binary/training_checkpoint.pt"),

        # Adım 2 için dinamik yollar
        "checkpoint_path_multiclass": os.path.join(output_dir, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir, "multiclass/best_model.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir, "multiclass/label_encoder_multiclass.pkl"),
        "resume_checkpoint_path_multiclass": os.path.join(output_dir, "multiclass/training_checkpoint.pt")
    }

    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    try:
        trial_log_file = os.path.join(output_dir, "trial_summary.log")
        setup_logging(trial_log_file)

        logging.info(f"\n--- DENEME {trial.number} BAŞLATILIYOR ({model_name}) ---")
        logging.info(f"Parametreler: {trial.params}")

        best_binary_val_f1 = train_top_level_classifier(config)
        best_multiclass_val_f1 = train_expert_classifier(config)

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

        logging.info(f"DENEME {trial.number} Sonuç: Binary Val F1: {best_binary_val_f1:.4f}, Multiclass Val F1: {best_multiclass_val_f1:.4f}")

        return best_multiclass_val_f1

    except Exception as e:
        try:
            trial_log_file = os.path.join(output_dir, "trial_summary.log")
            setup_logging(trial_log_file)
            logging.error(f"DENEME {trial.number} HATA ALDI: {e}", exc_info=True)
        except Exception as log_e:
            print(f"DENEME {trial.number} KRİTİK HATA: {e}")
            print(f"Loglama hatası: {log_e}")

        return 0.0  # Başarısız deneme


def print_dataset_info(model_name, data_path, seed):
    """
    Veri setlerini yükler, böler ve her bir bölümdeki sınıf dağılımlarını loglar.
    Eğitim yapmaz, sadece bilgi verir.
    """
    print("--- Veri Seti Dağılım İncelemesi Başlatılıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # log_class_distribution yardımcı fonksiyonunu train_v2.py'den alıyoruz
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

    # İkili ve Çok Sınıflı görevler için döngü
    for task in ['binary', 'multiclass']:
        print(f"\n{'=' * 20} GÖREV: {task.upper()} {'=' * 20}")

        # Veri setini ilgili görev için yükle
        full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=data_path, task=task)

        # Veriyi ayırma (Train/Val/Test)
        generator = Generator().manual_seed(seed)
        train_val_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=generator)

        train_size = int(0.85 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

        # Her bir set için dağılımı göster
        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Classifier Training with Optuna")
    parser.add_argument("--model_index", type=int, default=DEFAULT_MODEL_INDEX, help="Index of the model to train from MODEL_NAMES list.")
    args = parser.parse_args()
    model_index = args.model_index

    if DATASET_INFO:
        print_dataset_info(model_name=MODEL_NAMES[0], data_path=DATA_PATH, seed=42)
    else:

        try:
            model_name = MODEL_NAMES[model_index]
        except IndexError:
            print(f"HATA: Geçersiz model_index: {model_index}. Bu değer 0 ile {len(MODEL_NAMES) - 1} arasında olmalıdır.")
            return

        print(f"\n\n{'=' * 60}")
        print(f"--- BAŞLATILIYOR: {model_name} için {NUMBER_TRIALS} denemelik optimizasyon ---")
        print(f"{'=' * 60}\n")

        # Optuna çalışma dizini ve çalışma adı ayarları
        try:
            model_short_name = model_name.split('/')[-1]
            study_name = f"{model_short_name}_hiearchical_study"

            # .db dosyalarını da ana çıktı klasörüne kaydet
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
            print(f"En iyi değer (Best value - Uzman Model Macro F1): {study.best_value:.4f}")
            print("En iyi parametreler (Best params):")
            for key, value in study.best_params.items():
                print(f"    {key}: {value}")

            model_short_name = model_name.split('/')[-1]
            best_trial_dir = os.path.join(DATA_OUTPUT_PATH, f"trial_{study.best_trial.number}_{model_short_name}/")
            print(f"\nEn iyi modelin ve logların kaydedildiği klasör: {best_trial_dir}")

        except Exception as e:
            print(f"KRİTİK HATA: {model_name} için optimizasyon durduruldu. Hata: {e}")

        print(f"\n\n{'=' * 60}")
        print("TÜM MODELLERİN OPTİMİZASYONU TAMAMLANDI.")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()