import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import os
import logging
import json
import argparse  # Komut satırından argüman almak için

# Orijinal betiğinizdeki yardımcı sınıfları ve fonksiyonları import edin
from train_v1_1 import setup_logging, evaluate
from dataset import CitationDataset
from generic_model import TransformerClassifier


def continue_training(trial_dir, total_epochs):
    """
    Belirtilen bir deneme klasöründeki checkpoint'ten eğitimi devam ettirir.
    """
    # 1. Adım: Orijinal yapılandırmayı yükle
    config_path = os.path.join(trial_dir, "training_config.json")
    if not os.path.exists(config_path):
        print(f"HATA: {config_path} bulunamadı. Lütfen geçerli bir deneme klasörü belirtin.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Yeni toplam epoch sayısını config'e ekle
    config['total_epochs'] = total_epochs

    # Loglamayı aynı klasördeki log dosyasına ekleme yaparak devam ettir
    setup_logging(log_file=os.path.join(trial_dir, "training.log"))

    logging.info("==========================================================")
    logging.info(f"EĞİTİME DEVAM EDİLİYOR: {trial_dir}")
    logging.info(f"Yeni hedef epoch sayısı: {total_epochs}")
    logging.info("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])

    # 2. Adım: Tokenizer, Veri Seti ve DataLoader'ları yeniden oluştur
    # (Orijinal veri bölünmesini sağlamak için aynı seed kullanılır)
    tokenizer = AutoTokenizer.from_pretrained(trial_dir)  # Kaydedilmiş tokenizer'ı yükle
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'])

    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()

    generator = torch.Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, _ = random_split(full_dataset, [train_val_size, test_size], generator=generator)

    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # 3. Adım: Modeli, Optimizer'ı ve Scheduler'ı yeniden oluştur
    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    num_training_steps = len(train_loader) * total_epochs  # Yeni epoch sayısına göre ayarla
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()

    # 4. Adım: Checkpoint'i yükle
    checkpoint_path = os.path.join(trial_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        logging.error(f"HATA: {checkpoint_path} bulunamadı. Bu deneme hiç eğitilmemiş olabilir.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    logging.info(f"Checkpoint yüklendi. {start_epoch}. epoch'tan devam ediliyor.")

    if start_epoch >= total_epochs:
        logging.warning(
            f"Model zaten {start_epoch} epoch eğitilmiş. Yeni hedef epoch sayısı ({total_epochs}) daha düşük veya eşit. Eğitim yapılmayacak.")
        return

    # 5. Adım: Eğitim döngüsüne devam et
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=True)

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

        # ... (Değerlendirme ve kaydetme mantığı train_v1.py ile aynı)
        avg_train_loss = total_loss / len(train_loader)
        intent_val_acc, intent_report = evaluate(model, val_loader, device, label_names_list)
        logging.info(
            f"Epoch {epoch + 1} Tamamlandı. Ortalama Eğitim Kaybı: {avg_train_loss:.4f}, Doğrulama Başarımı: {intent_val_acc:.4f}")

        if intent_val_acc > best_val_acc:
            best_val_acc = intent_val_acc
            logging.info(f"🚀 Yeni en iyi doğrulama başarımı: {best_val_acc:.4f}. Model kaydediliyor...")
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pt"))

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_acc": best_val_acc
        }, checkpoint_path)

    logging.info("Eğitime devam etme işlemi tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir Optuna denemesinin eğitimine devam et.")
    parser.add_argument("--trial_dir", type=str, required=True,
                        help="Devam edilecek denemenin klasör yolu (örn: checkpoints_v1/bert-base-turkish-cased/trial_12/)")
    parser.add_argument("--total_epochs", type=int, required=True, help="Toplamda kaç epoch'a kadar eğitileceği.")

    args = parser.parse_args()

    continue_training(args.trial_dir, args.total_epochs)