from transformers import AutoTokenizer
from model_v1 import BerturkClassifier
import torch
import pickle
import os
from tqdm import tqdm

from dataset import CitationDataset
from torch.utils.data import random_split
from torch import Generator


class HierarchicalPredictor:
    # Bu sınıfta herhangi bir değişiklik yapmaya gerek yok.
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cihaz kullanılıyor: {self.device}")

        # --- Tokenizer'ı Yükle ---
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.max_len = 128

        # --- Label Encoder'ları Yükle ---
        with open(config["label_encoder_binary_path"], "rb") as f:
            self.le_binary = pickle.load(f)
        with open(config["label_encoder_multiclass_path"], "rb") as f:
            self.le_multiclass = pickle.load(f)

        # --- Modelleri Yükle ---
        # 1. Üst Seviye (İkili) Model
        num_labels_binary = len(self.le_binary.classes_)
        self.model_binary = BerturkClassifier(model_name=config["model_name"], num_labels=num_labels_binary)
        self.model_binary.bert.resize_token_embeddings(len(self.tokenizer))
        self.model_binary.load_state_dict(torch.load(config["model_binary_path"], map_location=self.device))
        self.model_binary.to(self.device)
        self.model_binary.eval()
        print("Üst seviye (ikili) model başarıyla yüklendi.")

        # 2. Uzman (Çok Sınıflı) Model
        num_labels_multiclass = len(self.le_multiclass.classes_)
        self.model_multiclass = BerturkClassifier(model_name=config["model_name"], num_labels=num_labels_multiclass)
        self.model_multiclass.bert.resize_token_embeddings(len(self.tokenizer))
        self.model_multiclass.load_state_dict(torch.load(config["model_multiclass_path"], map_location=self.device))
        self.model_multiclass.to(self.device)
        self.model_multiclass.eval()
        print("Uzman (çok sınıflı) model başarıyla yüklendi.")

    def predict(self, text):
        """
        Verilen bir metin için hiyerarşik sınıflandırma yapar.
        """
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits_binary = self.model_binary(input_ids, attention_mask)
            pred_binary_id = torch.argmax(logits_binary, dim=1).item()
            pred_binary_label = self.le_binary.classes_[pred_binary_id]

            if pred_binary_label.lower() == "non-background":
                logits_multiclass = self.model_multiclass(input_ids, attention_mask)
                pred_multiclass_id = torch.argmax(logits_multiclass, dim=1).item()
                final_label = self.le_multiclass.classes_[pred_multiclass_id]
            else:
                final_label = pred_binary_label
        return final_label


def main():
    config = {
        "model_name": "dbmdz/bert-base-turkish-cased",
        "model_binary_path": "checkpoints_v2/binary/best_model.pt",
        "model_multiclass_path": "checkpoints_v2/multiclass/best_model.pt",
        "label_encoder_binary_path": "checkpoints_v2/binary/label_encoder_binary.pkl",
        "label_encoder_multiclass_path": "checkpoints_v2/multiclass/label_encoder_multiclass.pkl",
        # GÜNCELLEME: train_v2.py ile aynı seed ve veri yolu eklendi
        "seed": 42,
        "csv_path": "data/data_v1.csv"
    }

    # GÜNCELLEME: Tahminciyi başlatmadan önce test verisini hazırlıyoruz
    print("--- Test Veri Seti Hazırlanıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Not: Orijinal metin ve etiketlere erişmek için task=None ile tüm veriyi yüklüyoruz.
    # Bölme işlemi aynı seed ile yapıldığı için sonuç değişmeyecektir.
    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config["csv_path"], task=None)

    # train_v2.py'deki ile BİREBİR AYNI bölme işlemini yap
    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=generator
    )

    print(f"Eğitimde kullanılan %20'lik test seti bulundu. Toplam {len(test_dataset)} örnek.")
    print("-" * 35)

    # Tahminciyi başlat
    predictor = HierarchicalPredictor(config)

    # GÜNCELLEME: Test seti üzerinde tahmin yap ve doğruluğu ölç
    print("\n--- TEST SETİ ÜZERİNDE TAHMİN SÜRECİ BAŞLATILIYOR ---")

    correct_predictions = 0

    # test_dataset'in içindeki her bir örnek için döngü kur
    for i in tqdm(range(len(test_dataset)), desc="Test Verileri İşleniyor"):
        # Orijinal index'i al
        original_idx = test_dataset.indices[i]

        # Orijinal DataFrame'den metni ve gerçek etiketi al
        text_to_predict = full_dataset.df.iloc[original_idx]['citation_context']
        true_label = full_dataset.df.iloc[original_idx]['citation_intent']

        # Tahmini yap
        predicted_label = predictor.predict(text_to_predict)

        # Sonuçları karşılaştır
        if predicted_label == true_label:
            correct_predictions += 1

    # Final doğruluğu hesapla ve yazdır
    accuracy = (correct_predictions / len(test_dataset)) * 100
    print("\n--- TAHMİN SONUÇLARI ---")
    print(f"Toplam Test Örneği: {len(test_dataset)}")
    print(f"Doğru Tahmin Sayısı: {correct_predictions}")
    print(f"Test Seti Başarımı (Accuracy): {accuracy:.2f}%")


if __name__ == "__main__":
    main()