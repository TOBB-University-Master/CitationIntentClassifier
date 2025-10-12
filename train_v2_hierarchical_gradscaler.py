import torch
import torch.nn as nn
import os
import logging
import pickle
import json
import optuna

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
from dataset import CitationDataset
from generic_model import TransformerClassifier
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI ***
# ==============================================================================
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
# MODEL_NAME = "dbmdz/electra-base-turkish-cased-discriminator"
# MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "microsoft/deberta-v3-base"

DATA_PATH = "data/data_v2.csv"

DATASET_INFO = True
# ==============================================================================



"""
     Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log) 
     hem de konsola yazdırmak için bir loglama sistemi kurar
"""
def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
    return acc, report

def display_samples(loader_name, data_loader, tokenizer, num_samples=1000):
    """
    Verilen bir DataLoader'dan belirtilen sayıda örneği yazdırır.
    """
    print(f"\n--- {loader_name} İçin {num_samples} Örnek Veri ---")

    # DataLoader'ı bir iteratöre dönüştür
    data_iter = iter(data_loader)

    # Örnekleri al ve yazdır
    for i in range(num_samples):
        try:
            sample = next(data_iter)

            # batch_size > 1 olabileceğinden her zaman batch'in ilk örneğini alıyoruz
            input_ids = sample['input_ids'][0]
            label = sample['label'][0]

            # Token ID'lerini tekrar okunabilir metne dönüştür
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

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
    """
    log_file = os.path.join(config["checkpoint_path_binary"], "training_binary.log")
    setup_logging(log_file)
    logging.info("--- ADIM 1: Üst Seviye (İkili) Sınıflandırıcı Eğitimi Başlatılıyor ---")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config["seed"])

    # Tokenizer (ortak)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Veri Seti (İkili görev için)
    logging.info("İkili sınıflandırma için veri seti yükleniyor...")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path="data/data_v1.csv", task='binary')
    num_labels = len(full_dataset.get_label_names())  # Bu 2 olmalı
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    # Label encoder'ı kaydet
    with open(config["label_encoder_binary_path"], "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"İkili label encoder şuraya kaydedildi: {config['label_encoder_binary_path']}")

    # Veriyi ayırma (Train/Val/Test)
    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )

    # 3. ADIM: %80'LİK KISMI %85 (TRAIN) VE %15 (VALIDATION) OLARAK AYIRMA
    logging.info("Eğitim/Doğrulama seti, %85 Eğitim ve %15 Doğrulama olarak ayrılıyor...")
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    if config["print_labels"]:
        label_names = full_dataset.label_encoder.classes_

        # Her bir veri setindeki (Subset) etiketleri sayan bir yardımcı fonksiyon
        # Her bir veri setindeki (Subset) etiketleri sayan bir yardımcı fonksiyon
        def log_class_distribution(subset, name):
            # DÜZELTME: Subset içindeki her bir elemanın etiketini doğrudan okuyoruz.
            # Bu yöntem, iç içe geçmiş Subset'lerde bile sorunsuz çalışır.
            labels = [subset[i]['label'].item() for i in range(len(subset))]

            counts = Counter(labels)
            label_names = subset.dataset.label_encoder.classes_ if not isinstance(subset.dataset,
                                                                                  Subset) else subset.dataset.dataset.label_encoder.classes_
            logging.info(f"--- {name} Sınıf Dağilımı ---")
            logging.info(f"Toplam Örnek: {len(subset)}")
            for label_id, count in sorted(counts.items()):
                # `label_names`'i doğru yerden aldığımızdan emin olalım
                original_dataset = subset
                while isinstance(original_dataset, Subset):
                    original_dataset = original_dataset.dataset

                label_names = original_dataset.label_encoder.classes_
                logging.info(f"    {label_names[label_id]} (ID: {label_id}): {count}")

        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")
        exit(0)

    # Model, Optimizer, Scheduler
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
    if os.path.exists(config["resume_checkpoint_path_binary"]):
        checkpoint = torch.load(config["resume_checkpoint_path_binary"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)  # Eski checkpoint'lerle uyumluluk için .get()
        logging.info(f"Checkpoint bulundu, eğitime {start_epoch}. epoch'tan devam ediliyor.")
    else:
        logging.info("Yeni bir eğitim başlatılıyor, checkpoint bulunamadı.")


    for epoch in range(start_epoch,config["epochs"]):
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
            "best_val_acc": best_val_acc
        }
        torch.save(checkpoint_data, config["resume_checkpoint_path_binary"])

        # Değerlendirme
        val_acc, val_report = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"\nEpoch {epoch + 1} - Doğrulama Başarımı: {val_acc:.4f}\n{val_report}")

        # En iyi modeli kaydetme
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi ikili model kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), config["best_model_path_binary"])

    # Eğitim sonrası test
    logging.info("\n--- İkili Model Test Süreci ---")
    model.load_state_dict(torch.load(config['best_model_path_binary']))
    test_acc, test_report = evaluate(model, test_loader, device, label_names_list)
    logging.info(f"\n--- İKİLİ TEST SONUÇLARI ---\nTest Başarımı: {test_acc:.4f}\n{test_report}")


# TODO: Uzman Sınıflandırıcı Eğitimi
def train_expert_classifier(config):
    """
    Modeli 'Non-Background' olan 4 sınıf üzerinde eğitir.
    """
    log_file = os.path.join(config["checkpoint_path_multiclass"], "training_multiclass.log")
    setup_logging(log_file)
    logging.info("\n--- ADIM 2: Uzman (Çok Sınıflı) Sınıflandırıcı Eğitimi Başlatılıyor ---")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config["seed"])

    # Tokenizer (ortak)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Veri Seti (Çok sınıflı görev için)
    logging.info("Çok sınıflı (Non-Background) veri seti yükleniyor...")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path="data/data_v1.csv", task='multiclass')
    num_labels = len(full_dataset.get_label_names())  # Bu 4 olmalı
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Sınıf sayısı: {num_labels}, Sınıflar: {label_names_list}")

    # Label encoder'ı kaydet
    with open(config["label_encoder_multiclass_path"], "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"Çok sınıflı label encoder şuraya kaydedildi: {config['label_encoder_multiclass_path']}")


    # Veriyi ayırma (Train/Val/Test)
    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )

    # 3. ADIM: %80'LİK KISMI %85 (TRAIN) VE %15 (VALIDATION) OLARAK AYIRMA
    logging.info("Eğitim/Doğrulama seti, %85 Eğitim ve %15 Doğrulama olarak ayrılıyor...")
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    if config["print_labels"]:
        label_names = full_dataset.label_encoder.classes_

        # Her bir veri setindeki (Subset) etiketleri sayan bir yardımcı fonksiyon
        def log_class_distribution(subset, name):
            # DÜZELTME: Subset içindeki her bir elemanın etiketini doğrudan okuyoruz.
            # Bu yöntem, iç içe geçmiş Subset'lerde bile sorunsuz çalışır.
            labels = [subset[i]['label'].item() for i in range(len(subset))]

            counts = Counter(labels)
            label_names = subset.dataset.label_encoder.classes_ if not isinstance(subset.dataset,
                                                                                  Subset) else subset.dataset.dataset.label_encoder.classes_
            logging.info(f"--- {name} Sınıf Dağilımı ---")
            logging.info(f"Toplam Örnek: {len(subset)}")
            for label_id, count in sorted(counts.items()):
                # `label_names`'i doğru yerden aldığımızdan emin olalım
                original_dataset = subset
                while isinstance(original_dataset, Subset):
                    original_dataset = original_dataset.dataset

                label_names = original_dataset.label_encoder.classes_
                logging.info(f"    {label_names[label_id]} (ID: {label_id}): {count}")

        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")
        exit(0)

    # Model, Optimizer, Scheduler
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
    if os.path.exists(config["resume_checkpoint_path_multiclass"]):
        checkpoint = torch.load(config["resume_checkpoint_path_multiclass"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)  # Eski checkpoint'lerle uyumluluk için .get()
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
            "best_val_acc": best_val_acc
        }
        torch.save(checkpoint_data, config["resume_checkpoint_path_multiclass"])

        # Değerlendirme
        val_acc, val_report = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"\nEpoch {epoch + 1} - Doğrulama Başarımı: {val_acc:.4f}\n{val_report}")

        # En iyi modeli kaydetme
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi uzman model kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), config["best_model_path_multiclass"])

    # Eğitim sonrası test
    logging.info("\n--- Uzman Model Test Süreci ---")
    model.load_state_dict(torch.load(config['best_model_path_multiclass']))
    test_acc, test_report = evaluate(model, test_loader, device, label_names_list)
    logging.info(f"\n--- UZMAN TEST SONUÇLARI ---\nTest Başarımı: {test_acc:.4f}\n{test_report}")


def main():
    #model_name = "dbmdz/bert-base-turkish-cased"
    #model_name = "dbmdz/electra-base-turkish-cased-discriminator"
    #model_name = "xlm-roberta-base"
    model_name = "microsoft/deberta-v3-base"

    model_short_name = model_name.split('/')[-1]
    output_dir = f"checkpoints_v2/{model_short_name}/"

    config = {
        "batch_size": 16,
        "epochs": 1,
        "lr": 2e-5,
        "model_name": model_name,
        "seed": 42,
        "print_labels": False,

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

    # Her bir adıma özel klasörlerin oluşturulduğundan emin ol
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    # İki adımı sırayla çalıştır
    train_top_level_classifier(config)
    train_expert_classifier(config)

    # TODO: Eğitim sonrası config kaydı...
    logging.info("\nEğitimler tamamlandı. Ortak yapılandırma dosyaları kaydediliyor...")

    # Tokenizer'ı (özel token ile birlikte) yeniden oluşturup kaydetmek,
    # her fonksiyonda ayrı ayrı tanımlandığı için en basit yoldur.
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # Kayıt için ortak bir ana klasör belirleyelim (örn: checkpoints_v2/)
    os.makedirs(output_dir, exist_ok=True)  # Klasör yoksa oluştur

    tokenizer.save_pretrained(output_dir)
    logging.info(f"Tokenizer dosyaları (vocab.txt, added_tokens.json vb.) şuraya kaydedildi: {output_dir}")

    # Eğitim yapılandırmasını JSON olarak kaydet
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logging.info(f"Yapılandırma dosyası şuraya kaydedildi: {config_path}")


def run_training_stage(config, trial, task_type):
    """
    Belirtilen görev için (binary veya multiclass) bir eğitim aşamasını çalıştırır.
    (Karma Hassasiyetli Eğitim - AMP ile güncellenmiştir)
    """
    is_binary = task_type == 'binary'
    task_name = "İkili" if is_binary else "Çok Sınıflı"

    # Dinamik olarak doğru yolları ve parametreleri seç
    output_dir = config["checkpoint_path_binary"] if is_binary else config["checkpoint_path_multiclass"]
    best_model_path = config["best_model_path_binary"] if is_binary else config["best_model_path_multiclass"]
    resume_checkpoint_path = config["resume_checkpoint_path_binary"] if is_binary else config[
        "resume_checkpoint_path_multiclass"]
    lr = config["lr_binary"] if is_binary else config["lr_multiclass"]
    epochs = config["epochs_binary"] if is_binary else config["epochs_multiclass"]

    log_file = os.path.join(output_dir, f"training_{task_type}.log")
    setup_logging(log_file)
    logging.info(f"--- Deneme #{trial.number} - {task_name} Sınıflandırıcı Eğitimi Başlatılıyor ---")

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

    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'], task=task_type)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, _ = random_split(full_dataset, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    start_epoch, best_val_acc = 0, 0.0

    # AMP >> GradScaler nesnesini oluştur. Sadece CUDA üzerinde çalışır.
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{epochs} ({task_name})")

        for batch in progress_bar:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
                batch["label"].to(device)

            optimizer.zero_grad()

            # AMP >> autocast ile ileri yayılım (forward pass) ve kayıp hesaplaması
            # Bu blok içindeki işlemler otomatik olarak FP16 formatında yapılır
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # AMP >> Kaybı ölçeklendirerek geriye yayılım (backward pass)
            scaler.scale(loss).backward()

            # AMP >> Optimizer adımını at
            scaler.step(optimizer)

            # AMP >> Bir sonraki iterasyon için scaler'ı güncelle
            scaler.update()

            lr_scheduler.step()
            total_loss += loss.item()

        val_acc, _ = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"Epoch {epoch + 1} - {task_type} Doğrulama Başarımı: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi {task_type} model kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), best_model_path)

    logging.info(f"--- {task_type} Sınıflandırıcı Eğitimi Tamamlandı ---")


def evaluate_hierarchical(config):
    """
    Eğitilmiş ikili ve uzman modellerle hiyerarşik birleşik performansı ölçer.
    """
    logging.info("\n--- Birleşik Hiyerarşik Değerlendirme Başlatılıyor ---")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ÖNEMLİ: Değerlendirme için tüm sınıfları içeren orijinal veri setini kullan
    full_dataset_orig = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'], task='all')

    # İkili ve Çok Sınıflı görevlerin label encoder'larını yükle
    with open(config["label_encoder_binary_path"], "rb") as f:
        binary_encoder = pickle.load(f)
    with open(config["label_encoder_multiclass_path"], "rb") as f:
        multiclass_encoder = pickle.load(f)

    # İkili modelin "Non-Background" etiketinin ID'sini bul
    non_background_binary_id = list(binary_encoder.transform(['Non-Background']))[0]

    # Modelleri oluştur ve eğitilmiş en iyi ağırlıkları yükle
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

    # Orijinal veri setinden doğrulama (validation) setini ayır
    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset_orig))
    test_size = len(full_dataset_orig) - train_val_size
    train_val_dataset, _ = random_split(full_dataset_orig, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    _, val_dataset_orig = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    val_loader_orig = DataLoader(val_dataset_orig, batch_size=config["batch_size"])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader_orig:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)

            # Adım 1: Üst seviye model ile tahmin yap
            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.zeros_like(binary_preds)

            # Adım 2: Uzman modele danışılacak verileri belirle
            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                # Sadece uzmanlık gerektiren input'ları seç
                expert_input_ids = input_ids[expert_indices]
                expert_attention_mask = attention_mask[expert_indices]

                # Uzman model ile tahmin yap
                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)

                # Uzman modelin tahminlerini (0,1,2,3) orijinal etiketlere dönüştür
                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = full_dataset_orig.label_encoder.transform(multiclass_class_names)

                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            # İkili modelin "Background" dediği verilerin etiketini de ekle (ID: 0)
            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            background_orig_id = full_dataset_orig.label_encoder.transform(['Background'])[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı: {overall_accuracy:.4f}")
    return overall_accuracy


def objective(trial):
    model_short_name = MODEL_NAME.split('/')[-1]
    output_dir_base = f"checkpoints_v2/{model_short_name}/trial_{trial.number}/"

    config = {
        "data_path": DATA_PATH,
        "model_name": MODEL_NAME,
        "seed": 42,

        # Denenecek Hiperparametreler
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr_binary": trial.suggest_float("lr_binary", 1e-5, 5e-5, log=True),
        "lr_multiclass": trial.suggest_float("lr_multiclass", 1e-5, 5e-5, log=True),
        "epochs_binary": trial.suggest_int("epochs_binary", 2, 5),
        "epochs_multiclass": trial.suggest_int("epochs_multiclass", 5, 15),

        # Yollar
        "checkpoint_path_binary": os.path.join(output_dir_base, "binary/"),
        "best_model_path_binary": os.path.join(output_dir_base, "binary/best_model.pt"),
        "resume_checkpoint_path_binary": os.path.join(output_dir_base, "binary/training_checkpoint.pt"),
        "label_encoder_binary_path": os.path.join(output_dir_base, "binary/label_encoder.pkl"),

        "checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir_base, "multiclass/best_model.pt"),
        "resume_checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/training_checkpoint.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir_base, "multiclass/label_encoder.pkl"),
    }
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    # 1. Aşama: İkili Modeli Eğit
    run_training_stage(config, trial, 'binary')

    # 2. Aşama: Uzman Modeli Eğit
    run_training_stage(config, trial, 'multiclass')

    # 3. Aşama: İki modelin ortak performansını ölç
    overall_accuracy = evaluate_hierarchical(config)

    # Optuna'ya optimize edeceği değeri döndür
    return overall_accuracy


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

if __name__ == "__main__":
    if DATASET_INFO:
        print_dataset_info(model_name=MODEL_NAME, data_path=DATA_PATH, seed=42)
    else:
        model_short_name = MODEL_NAME.split('/')[-1]
        study_name = f"{model_short_name}_hiearchical_study"
        storage_path = f"sqlite:///{model_short_name}_hierarchical.db"

        print(f"🚀 Hiyerarşik Optimizasyon Başlatılıyor 🚀")
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

        # Hiyerarşik model daha karmaşık olduğu için deneme sayısını artırmak iyi olabilir
        study.optimize(objective, n_trials=50)

        print("\nOptimizasyon tamamlandı.")
        print("En iyi deneme:")
        trial = study.best_trial
        print(f"  Değer (En Yüksek Birleşik Doğruluk): {trial.value}")
        print("  En İyi Parametreler: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")