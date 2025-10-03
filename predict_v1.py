import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
import pandas as pd
import pickle
import json
import os
import logging
from tqdm import tqdm
from torch import Generator
from sklearn.metrics import classification_report, accuracy_score

from generic_model import TransformerClassifier
from dataset import CitationDataset

"""
    Tahmin sürecindeki önemli bilgileri hem bir dosyaya (prediction.log)
    hem de konsola yazdırmak için bir loglama sistemi kurar.
"""
def setup_logging(log_file):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])


def main():
    #model_dir = "checkpoints_v1/bert-base-turkish-cased"
    #model_dir = "checkpoints_v1/electra-base-turkish-cased-discriminator"
    #model_dir = "checkpoints_v1/xlm-roberta-base"
    model_dir = "checkpoints_v1/deberta-v3-base"

    # Loglama sistemini ayarla
    setup_logging(os.path.join(model_dir, "prediction.log"))

    # 1. Gerekli dosyaların yollarını ve yapılandırmayı yükle
    logging.info("--- TAHMİN SÜRECİ BAŞLATILIYOR ---")
    try:
        config_path = os.path.join(model_dir, "training_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info("Eğitim yapılandırması (training_config.json) yüklendi.")

        best_model_path = os.path.join(model_dir, "best_model.pt")
        label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

        # Dosyaların varlığını kontrol et
        if not os.path.exists(best_model_path) or not os.path.exists(label_encoder_path):
            raise FileNotFoundError("Gerekli model veya label encoder dosyaları bulunamadı.")

    except FileNotFoundError as e:
        logging.error(f"Hata: {e}")
        logging.error(
            f"Lütfen '{model_dir}' dizininin doğru olduğundan ve içinde 'best_model.pt', 'label_encoder.pkl' ve 'training_config.json' dosyalarının bulunduğundan emin olun.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    logging.info(f"Cihaz seçildi: {device}")

    # 2. Tokenizer ve Label Encoder'ı yükle
    logging.info(f"Tokenizer yükleniyor: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    logging.info("Label Encoder yükleniyor...")
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    label_names_list = label_encoder.classes_.tolist()
    num_labels = len(label_names_list)
    logging.info(f"Toplam {num_labels} etiket yüklendi: {label_names_list}")

    # 3. Veri Setini Yükle ve Eğitimdekiyle Aynı Şekilde Böl
    logging.info("Ana veri seti yükleniyor: data/data_v1.csv")
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path="data/data_v1.csv")

    # Eğitimdeki random_split'i TEKRARLAMAK için aynı seed ile generator oluştur
    generator = Generator().manual_seed(config["seed"])

    logging.info(f"Veri seti, eğitimdekiyle aynı seed ({config['seed']}) kullanılarak yeniden bölünüyor...")
    # 2. ADIM: VERİYİ %80 (TRAIN+VAL) VE %20 (TEST) OLARAK AYIRMA
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )
    logging.info(f"Test seti ayrıldı. Toplam {len(test_dataset)} örnek.")

    # Test verileri için DataLoader oluştur
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # 4. Modeli Yükle
    logging.info("Model mimarisi oluşturuluyor...")
    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)


    # Tokenizer'ın boyutuna göre modelin embedding katmanını yeniden boyutlandır.
    # Bu, yükleyeceğimiz ağırlıklarla modelin mimarisini eşleştirecektir.
    logging.info(
        f"Modelin token embedding katmanı, tokenizer'ın boyutu olan {len(tokenizer)}'e göre yeniden boyutlandırılıyor.")
    model.transformer.resize_token_embeddings(len(tokenizer))
    # --------------------

    logging.info(f"Eğitilmiş en iyi model yükleniyor: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()  # Modeli değerlendirme moduna al

    # 5. Tahmin Döngüsü
    logging.info("Test verileri üzerinde tahminler yapılıyor...")
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Gradyan hesaplamalarını kapat
        for batch in tqdm(test_loader, desc="Tahmin ediliyor"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Sonuçları Değerlendir ve Kaydet
    logging.info("Tahminler tamamlandı. Sonuçlar işleniyor...")

    # Tahmin edilen ID'leri ve gerçek ID'leri etiket isimlerine dönüştür
    predicted_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)

    # Orijinal metinleri test setinden al
    test_indices = test_dataset.indices
    original_texts = full_dataset.df.iloc[test_indices]['citation_context'].tolist()

    # Sonuçları bir DataFrame'e kaydet
    results_df = pd.DataFrame({
        'citation_context': original_texts,
        'true_label': true_labels,
        'predicted_label': predicted_labels
    })

    # Sonuçları CSV dosyasına yaz
    results_path = os.path.join(model_dir, "test_predictions.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"Tahmin sonuçları '{results_path}' dosyasına kaydedildi.")

    # Performans metriklerini hesapla ve göster
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels, zero_division=0)

    logging.info(f"\n--- TEST SONUÇLARI ---")
    logging.info(f"Test Başarımı (Accuracy): {accuracy:.4f}")
    logging.info(f"Test Sınıflandırma Raporu:\n{report}")
    logging.info("İşlem bitti.")


if __name__ == "__main__":
    main()