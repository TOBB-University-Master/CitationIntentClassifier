import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from torch import Generator
from torch.utils.data import DataLoader, random_split
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW

from dataset import CitationDataset
from model import BerturkClassifier
from tqdm import tqdm
import os
import logging
import json
import pickle  # LabelEncoder ve SectionEncoder'ı kaydetmek için

"""
     Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log) 
     hem de konsola yazdırmak için bir loglama sistemi kurar
"""
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("checkpoints/training.log", mode='a'),
            logging.StreamHandler()
        ]
    )



"""
    Modeli değerlendirir ve doğruluk ile sınıflandırma raporu döndürür.
"""
def evaluate(model, data_loader, device, label_names):
    model.eval()
    all_intent_preds = []
    all_intent_labels = []
    # Section tahminine dair değişkenler kaldırıldı

    # Gradyan hesaplamalarını kapatır
    # Değerlendirme yapılırken modelin ağırlıkları güncellenmez
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)
            section_ids = batch["section_id"].to(device)  # section_id'ler batch'ten alındı

            # Modele section_ids'ler de girdi olarak verildi
            intent_logits = model(input_ids, attention_mask, section_ids)
            intent_preds = torch.argmax(intent_logits, dim=1)

            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())
            # Section ile ilgili tahminler ve metrikler kaldırıldı

    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
    intent_report = classification_report(all_intent_labels, all_intent_preds, target_names=label_names,
                                          zero_division=0)

    # Sadece intent doğruluğu ve raporu döndürüldü
    return intent_acc, intent_report

    """
        Args:
            section_embed_dim (int): Tahmin edilecek section için embedding uzunluğu eklenmiştir
    """
def main():
    config = {
        "batch_size": 16,
        "epochs": 20,
        "lr": 2e-5,
        "model_name": "dbmdz/bert-base-turkish-cased",
        "checkpoint_path": "checkpoints/berturk_classifier_checkpoint.pt",
        "best_model_path": "checkpoints/best_model.pt",
        "seed": 42,
        "section_embed_dim": 50  # section embedding boyutu
    }

    os.makedirs("checkpoints", exist_ok=True)
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    logging.info(f"Cihaz seçildi: {device}")

    # Tokenizer
    logging.info(f"Tokenizer yükleniyor: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Dataset ve DataLoader
    # CitationDataset'in tokenizer'ı __init__'te alacak şekilde güncellendiğini varsayıyoruz.
    dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path="data/train.csv")

    num_labels = len(dataset.get_label_names())
    num_sections = len(dataset.get_section_names())
    logging.info(f"Toplam atıf niyeti sınıfı: {num_labels}")
    logging.info(f"Toplam bölüm sınıfı: {num_sections}")  # Bölüm sayısını logla

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = Generator().manual_seed(config["seed"])

    # ------------------------------------------------------------
    # TODO: Burada test için ayrılan data hep aynı mı kaldı ?
    # ------------------------------------------------------------
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    logging.info(f"Veri seti yüklendi. Eğitim: {len(train_dataset)} örnek, Doğrulama: {len(val_dataset)} örnek.")

    # Model, Optimizer, Scheduler
    # Modele num_sections ve section_embed_dim parametreleri verildi
    model = BerturkClassifier(model_name=config["model_name"],
                              num_labels=num_labels,
                              num_sections=num_sections,
                              section_embed_dim=config["section_embed_dim"])

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    num_training_steps = len(train_loader) * config["epochs"]
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    # Checkpoint kontrol
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists(config["checkpoint_path"]):
        checkpoint = torch.load(config["checkpoint_path"], map_location=device)
        # DİKKAT: Eğer önceki checkpoint eski model yapısından (iki çıktı) ise,
        # bu model.load_state_dict() çağrısı hata verebilir.
        # Bu durumda eski checkpoint dosyasını silip yeni eğitime başlamanız gerekebilir.
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        logging.info(f"Checkpoint yüklendi, {start_epoch}. epoch'tan devam ediliyor.")
    else:
        logging.info("Yeni model eğitimi başlatılıyor.")

    # Eğitim Döngüsü
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)
            section_ids = batch["section_id"].to(device)  # section_id'ler alındı

            # Modele section_ids'ler de girdi olarak verildi
            intent_logits = model(input_ids, attention_mask, section_ids)

            # Sadece intent kaybı hesaplandı
            # CrossEntropyLoss()
            loss = criterion(intent_logits, intent_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1} Tamamlandı. Ortalama Eğitim Kaybı: {avg_train_loss:.4f}")

        # Doğrulama
        # evaluate çağrısı ve sonuçların kullanımı güncellendi
        intent_val_acc, intent_report = evaluate(
            model, val_loader, device, dataset.get_label_names()
        )
        logging.info(f"Doğrulama Başarımı (Intent Accuracy): {intent_val_acc:.4f}")
        logging.info(f"Intent Sınıflandırma Raporu:\n{intent_report}")
        # Section ile ilgili loglar kaldırıldı

        # Checkpoint'i her zaman kaydet
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_acc": best_val_acc  # Bu hala intent acc'yi takip ediyor
        }
        torch.save(checkpoint_data, config["checkpoint_path"])

        # Sadece en iyi modeli ayrı bir dosyaya kaydet (intent accuracy'ye göre)
        if intent_val_acc > best_val_acc:
            best_val_acc = intent_val_acc
            logging.info(f"🚀 Yeni en iyi doğrulama başarımı (Intent): {best_val_acc:.4f}. En iyi model kaydediliyor...")
            torch.save(model.state_dict(), config["best_model_path"])

    # Eğitim sonrası kayıt
    logging.info("\nEğitim tamamlandı.")
    logging.info("En son tokenizer yapılandırması kaydediliyor...")
    tokenizer.save_pretrained(os.path.dirname(config["best_model_path"]))

    # LabelEncoder ve SectionEncoder'ı da kaydet (tahmin aşamasında lazım olacak)
    with open(os.path.join(os.path.dirname(config["best_model_path"]), "label_encoder.pkl"), "wb") as f:
        pickle.dump(dataset.label_encoder, f)
    with open(os.path.join(os.path.dirname(config["best_model_path"]), "section_encoder.pkl"), "wb") as f:
        pickle.dump(dataset.section_encoder, f)

    # Config'i de kaydet
    config_path = os.path.join(os.path.dirname(config["best_model_path"]), "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    logging.info("İşlem bitti.")


if __name__ == "__main__":
    main()
