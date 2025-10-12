import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from torch import Generator
from torch.utils.data import DataLoader, random_split
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from dataset import CitationDataset
from generic_model import TransformerClassifier
from tqdm import tqdm
import os
import logging
import json
import pickle  # LabelEncoder ve SectionEncoder'ı kaydetmek için
import optuna   # Optuna kütüphanesi hiperparametre optimizasyonu için

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI ***
# ==============================================================================
# MODEL_NAME = "dbmdz/bert-base-turkish-cased"
# MODEL_NAME = "dbmdz/electra-base-turkish-cased-discriminator"
 #MODEL_NAME = "xlm-roberta-base"
MODEL_NAME = "microsoft/deberta-v3-base"

DATA_PATH = "data/data_v2.csv"
# ==============================================================================


"""
     Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log) 
     hem de konsola yazdırmak için bir loglama sistemi kurar
"""
def setup_logging(log_file):
    # Log dosyasını dinamik olarak ayarla
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
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

    # Gradyan hesaplamalarını kapatır
    # Değerlendirme yapılırken modelin ağırlıkları güncellenmez
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)

            # Modele section_ids'ler de girdi olarak verildi
            intent_logits = model(input_ids, attention_mask)
            intent_preds = torch.argmax(intent_logits, dim=1)

            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_labels.cpu().numpy())

    intent_acc = accuracy_score(all_intent_labels, all_intent_preds)
    intent_report = classification_report(all_intent_labels,
                                          all_intent_preds,
                                          target_names=label_names,
                                          zero_division=0)

    # Sadece intent doğruluğu ve raporu döndürüldü
    return intent_acc, intent_report

    """
        Args:
            section_embed_dim (int): Tahmin edilecek section için embedding uzunluğu eklenmiştir
    """
def objective(trial):
    config = {
        "data_path": DATA_PATH,
        "model_name": MODEL_NAME,
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "epochs": trial.suggest_int("epochs", 5,30),
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "seed": 42
    }

    # Model adına göre dinamik çıktı klasörü oluştur
    model_short_name = config["model_name"].split('/')[-1]
    output_dir = f"checkpoints_v1/{model_short_name}/trial_{trial.number}/"
    os.makedirs(output_dir, exist_ok=True)

    # Dinamik dosya yolları
    config["checkpoint_path"] = os.path.join(output_dir, "checkpoint.pt")
    config["best_model_path"] = os.path.join(output_dir, "best_model.pt")

    setup_logging(log_file=os.path.join(output_dir, "training.log"))

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    logging.info(f"Kullanılan Model: {config['model_name']}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Veriyi yükle ve hazırla
    logging.info("Ana veri seti yükleniyor: data/data_v1.csv")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'])
    logging.info(f"Toplam kayıt sayısı: {len(full_dataset)}")

    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()
    logging.info(f"Toplam atıf niyeti sınıfı: {num_labels}")

    # Tekrarlanabilirliği sağlamak için jeneratörü ayarla
    generator = Generator().manual_seed(config["seed"])

    # VERİYİ %80 (TRAIN+VAL) VE %20 (TEST) OLARAK AYIRMA
    logging.info("Veri seti, %80 Eğitim/Doğrulama ve %20 Test olarak ayrılıyor...")
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

    logging.info(f"Veri seti yüklendi. Eğitim: {len(train_dataset)} örnek, Doğrulama: {len(val_dataset)} örnek.")

    # Model, Optimizer, Scheduler
    # Modele num_sections parametreleri verildi
    model = TransformerClassifier(model_name=config["model_name"],
                              num_labels=num_labels)

    # Adım 1: Tokenizer'a yeni özel token'ı ekle
    logging.info("Tokenizer'a <CITE> token'ı ekleniyor.")
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # Adım 2: Modelin embedding katmanını yeni token sayısına göre yeniden boyutlandır
    logging.info("Modelin token embedding katmanı yeniden boyutlandırılıyor.")
    model.transformer.resize_token_embeddings(len(tokenizer))
    # ------------------------------------------------------------

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
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

        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{config['epochs']}", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent_labels = batch["label"].to(device)

            # Modele section_ids'ler de girdi olarak verildi
            intent_logits = model(input_ids, attention_mask)

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

        intent_val_acc, intent_report = evaluate(
            model, val_loader, device, label_names_list
        )
        logging.info(f"Doğrulama Başarımı (Intent Accuracy): {intent_val_acc:.4f}")
        logging.info(f"Intent Sınıflandırma Raporu:\n{intent_report}")

        # Sadece en iyi modeli ayrı bir dosyaya kaydet (intent accuracy'ye göre)
        if intent_val_acc > best_val_acc:
            best_val_acc = intent_val_acc
            logging.info(f"🚀 Yeni en iyi doğrulama başarımı (Intent): {best_val_acc:.4f}. En iyi model kaydediliyor...")
            torch.save(model.state_dict(), config["best_model_path"])

        # Checkpoint'i her zaman kaydet
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_val_acc": best_val_acc
        }, config["checkpoint_path"])

    # Eğitim sonrası kayıt
    logging.info("\nEğitim tamamlandı.")
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=4)


    logging.info(f"--- Deneme #{trial.number} Tamamlandı. En İyi Doğrulama Başarımı: {best_val_acc:.4f} ---")

    return best_val_acc

'''
    ### DEĞİŞİKLİK 5: Eğitim sonunda otomatik test adımı eklendi ###
    logging.info("\n--- NİHAİ TEST SÜRECİ BAŞLATILIYOR ---")
    logging.info(f"En iyi model yükleniyor: {config['best_model_path']}")
    model.load_state_dict(torch.load(config['best_model_path']))
    model.to(device)

    logging.info("Test veri seti üzerinde değerlendirme yapılıyor...")
    test_acc, test_report = evaluate(
        model, test_loader, device, label_names_list
    )
    logging.info(f"\n--- TEST SONUÇLARI ---")
    logging.info(f"Test Başarımı (Accuracy): {test_acc:.4f}")
    logging.info(f"Test Sınıflandırma Raporu:\n{test_report}")

    logging.info("İşlem bitti.")
'''

if __name__ == "__main__":

    model_short_name = MODEL_NAME.split('/')[-1]
    study_name = f"{model_short_name}_study"
    storage_path = f"sqlite:///{model_short_name}.db"

    print(f"🚀 Optimizasyon Başlatılıyor 🚀")
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

    # n_trials: Toplamda kaç farklı parametre kombinasyonu deneneceğini belirtir
    study.optimize(objective, n_trials=20)

    print("Optimizasyon tamamlandı.")
    print("En iyi deneme:")
    trial = study.best_trial

    print(f"  Değer (En Yüksek Validation Accuracy): {trial.value}")
    print("  En İyi Parametreler: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

