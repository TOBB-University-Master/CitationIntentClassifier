import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
import warnings

# --- 1. Sabitler ---

# Değerlendirilecek 'one_hot' test dosyası
CSV_FILE_PATH = "data/data_v2_test_one_hot.csv"

MODEL_PREFIXES = [
    "gemini-flash-k0",
    "gemini-flash-k1",
    "gemini-flash-k2",
    "gemini-flash-k5",
    "chatgpt-4o-k0",
    "chatgpt-4o-k1",
    "chatgpt-4o-k2",
    "chatgpt-4o-k5",
    "dspy",
    "gemini-pro-k0"
]

# Sütun sırasına göre etiketler (Kullanıcı tarafından belirtildiği gibi)
LABELS = ['background', 'basis', 'support', 'differ', 'discuss']


def calculate_majority_vote_accuracy():
    """
    'one_hot.csv' dosyasındaki tüm modellerin tahminlerini kullanarak
    çoğunluk oylaması (majority voting) yapar ve başarı metriklerini hesaplar.
    """
    try:
        # --- 1. Ground Truth (Gerçek Değerler) ve Tahminleri Yükleme ---
        print(f"'{CSV_FILE_PATH}' dosyasından veri yükleniyor...")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Test verisi yüklendi. Toplam {len(df)} kayıt.")

        # Gerçek etiketleri (y_true) al
        y_true = df['true_label']

        # Nihai tahminleri (çoğunluk oylaması sonucu) tutacak liste
        y_pred = []

        print(f"Toplam {len(MODEL_PREFIXES)} model üzerinden çoğunluk oylaması hesaplanıyor...")

        # --- 2. Her Satır İçin Çoğunluk Oylaması Hesaplama ---
        # tqdm ile ilerleme çubuğu
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Oylama Hesaplanıyor"):

            # Bu satır (citation) için modellerin oylarını tutan liste
            votes = []

            # Her bir modelin oyunu bul
            for prefix in MODEL_PREFIXES:
                model_voted = False
                for label in LABELS:
                    col_name = f"{prefix}_{label}"

                    # Modelin bu etiket için '1' verip vermediğini kontrol et
                    if row[col_name] == 1:
                        votes.append(label)
                        model_voted = True
                        break  # Bu modelin oyunu bulduk, sonraki modele geç

                if not model_voted:
                    # Bu durum, modelin o satır için (örn: 'dspy') bir tahmin üretmediği
                    # veya one-hot vektörde bir hata olduğu anlamına gelebilir.
                    # Oylamaya katılmaması için 'None' ekleyebiliriz (opsiyonel)
                    pass

                    # Oyları topla ve en çok oyu alanı (mod) bul
            if not votes:
                # Eğer hiçbir model oy vermemişse (imkansız gibi ama önlem)
                y_pred.append(None)  # veya varsayılan bir etiket
            else:
                # pandas'ın mode() fonksiyonu en sık kullanılanı bulur.
                # Eşitlik durumunda (örn: 5 'background', 5 'basis') ilkini alır ([0]).
                majority_vote = pd.Series(votes).mode()[0]
                y_pred.append(majority_vote)

        # --- 3. Metrikleri Hesaplama ve Sonuçları Görüntüleme ---
        print("\nOylama tamamlandı. Metrikler hesaplanıyor...")

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        report_str = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

        print("\n" + "=" * 50)
        print("LLM Ensemble - Majority Voting Sonuçları")
        print("=" * 50)
        print(f"Model Doğruluğu (Accuracy): {accuracy:.4f}")
        print(f"Macro F1 Skoru:             {macro_f1:.4f}")
        print("\nSınıflandırma Raporu:")
        print(report_str)
        print("=" * 50)

    except FileNotFoundError:
        print(f"HATA: '{CSV_FILE_PATH}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except KeyError as e:
        print(f"\nHATA: CSV dosyasında beklenen bir sütun bulunamadı: {e}")
        print("Lütfen 'MODEL_PREFIXES' veya 'LABELS' listesinin güncel olduğundan emin olun.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    # Pandas'ın .mode() ile ilgili olası bir Future uyarsını bastır
    warnings.simplefilter(action='ignore', category=FutureWarning)

    calculate_majority_vote_accuracy()