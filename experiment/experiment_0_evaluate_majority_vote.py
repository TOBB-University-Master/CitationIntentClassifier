import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import warnings

# --- 1. Sabitler ---

# Değerlendirilecek dosya yolu (Kendi dosya yolunuzla güncelleyin)
CSV_FILE_PATH = "../data/data_v2_test_one_hot_104.csv"

# Sütun sonlarında arayacağımız etiketler
LABELS = ['background', 'basis', 'support', 'differ', 'discuss']


def discover_model_prefixes(columns, labels):
    """
    Sütun isimlerini tarar ve etiketlerden (labels) yola çıkarak
    benzersiz model prefix'lerini bulur.

    Örnek: 'bert-base_1_background' sütunu için prefix 'bert-base_1' olarak belirlenir.
    """
    prefixes = set()

    for col in columns:
        for label in labels:
            # Sütun ismi _{label} ile bitiyorsa (örn: _background)
            suffix = f"_{label}"
            if col.endswith(suffix):
                # Suffix'i çıkarıp geri kalanı prefix olarak alıyoruz
                prefix = col.rsplit(suffix, 1)[0]
                prefixes.add(prefix)

    # Sıralı liste döndür (tutarlılık için)
    return sorted(list(prefixes))


def calculate_majority_vote_accuracy():
    """
    Dosyadaki modelleri dinamik olarak bulur, çoğunluk oylaması (majority voting) yapar
    ve başarı metriklerini hesaplar.
    """
    try:
        # --- 1. Veri Yükleme ve Model Tespiti ---
        print(f"'{CSV_FILE_PATH}' dosyasından veri yükleniyor...")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Test verisi yüklendi. Toplam {len(df)} kayıt.")

        # Sütun isimlerinden model prefixlerini otomatik çıkar
        model_prefixes = discover_model_prefixes(df.columns, LABELS)

        if not model_prefixes:
            print("HATA: Belirtilen etiketlere uygun hiçbir model sütunu bulunamadı.")
            return

        print(f"\nTespit edilen modeller ({len(model_prefixes)} adet):")
        for mp in model_prefixes:
            print(f" - {mp}")

        # Gerçek etiketleri (y_true) al
        # Not: CSV'de 'true_label' adında bir sütun olduğu varsayılıyor.
        if 'true_label' not in df.columns:
            raise KeyError("CSV dosyasında 'true_label' sütunu bulunamadı.")

        y_true = df['true_label']
        y_pred = []

        print(f"\nÇoğunluk oylaması hesaplanıyor...")

        # --- 2. Her Satır İçin Çoğunluk Oylaması Hesaplama ---
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Oylama İlerleyişi"):
            votes = []

            # Dinamik olarak bulunan her model için tahminleri kontrol et
            for prefix in model_prefixes:
                model_voted = False
                for label in LABELS:
                    # Sütun ismini oluştur: prefix + "_" + label
                    col_name = f"{prefix}_{label}"

                    # Sütun var mı kontrolü (opsiyonel ama güvenli)
                    if col_name in row and row[col_name] == 1:
                        votes.append(label)
                        model_voted = True
                        break

                        # Eğer modelin o satırda hiç 1'i yoksa (tahmin yapamamışsa) pas geçiyoruz.

            # Oyları topla
            if not votes:
                y_pred.append(None)  # Hiçbir model oy kullanmadıysa
            else:
                # En çok tekrar eden etiketi (mode) al
                majority_vote = pd.Series(votes).mode()[0]
                y_pred.append(majority_vote)

        # --- 3. Metrikleri Hesaplama ---
        # None değerleri varsa temizleyelim (veya hata fırlatalım, burada filtreliyoruz)
        valid_indices = [i for i, x in enumerate(y_pred) if x is not None]

        if len(valid_indices) < len(y_pred):
            print(
                f"\nUYARI: {len(y_pred) - len(valid_indices)} satır için çoğunluk kararı verilemedi (tahmin yok). Bu satırlar atlanıyor.")
            filtered_y_true = y_true.iloc[valid_indices]
            filtered_y_pred = [y_pred[i] for i in valid_indices]
        else:
            filtered_y_true = y_true
            filtered_y_pred = y_pred

        accuracy = accuracy_score(filtered_y_true, filtered_y_pred)
        macro_f1 = f1_score(filtered_y_true, filtered_y_pred, average='macro', zero_division=0)
        report_str = classification_report(filtered_y_true, filtered_y_pred, labels=LABELS, zero_division=0)

        print("\n" + "=" * 50)
        print("Otomatik Algılanan Modeller ile Majority Voting Sonuçları")
        print("=" * 50)
        print(f"Model Doğruluğu (Accuracy): {accuracy:.4f}")
        print(f"Macro F1 Skoru:             {macro_f1:.4f}")
        print("\nSınıflandırma Raporu:")
        print(report_str)
        print("=" * 50)

    except FileNotFoundError:
        print(f"HATA: '{CSV_FILE_PATH}' dosyası bulunamadı.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    calculate_majority_vote_accuracy()