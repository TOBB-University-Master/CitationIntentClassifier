import torch
import torch.nn as nn
import os
import logging
import pickle
import json

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
from dataset import CitationDataset
from model_v1 import BerturkClassifier
from tqdm import tqdm


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = BerturkClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.bert.resize_token_embeddings(len(tokenizer))
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = BerturkClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.bert.resize_token_embeddings(len(tokenizer))
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
    output_dir = "checkpoints_v2/"
    config = {
        "batch_size": 16,
        "epochs": 30,  # Her adım için epoch sayısı
        "lr": 2e-5,
        "model_name": "dbmdz/bert-base-turkish-cased",
        "seed": 42,
        "print_labels": False,

        # Adım 1 için yollar
        "checkpoint_path_binary": output_dir + "binary/",
        "best_model_path_binary": output_dir + "binary/best_model.pt",
        "label_encoder_binary_path": output_dir + "binary/label_encoder_binary.pkl",
        "resume_checkpoint_path_binary": output_dir + "binary/training_checkpoint.pt",

        # Adım 2 için yollar
        "checkpoint_path_multiclass": output_dir + "multiclass/",
        "best_model_path_multiclass": output_dir + "multiclass/best_model.pt",
        "label_encoder_multiclass_path": output_dir + "multiclass/label_encoder_multiclass.pkl",
        "resume_checkpoint_path_multiclass": output_dir + "multiclass/training_checkpoint.pt"
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


if __name__ == "__main__":
    main()