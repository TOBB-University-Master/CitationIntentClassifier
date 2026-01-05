import pandas as pd
import joblib
import os
import argparse
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
    Meta-Model Değerlendirme Betiği
    ---------------------------------
    Bu betik, önceden eğitilmiş lojistik regresyon (lr) veya XGBoost (xgb) meta-modellerini
    yükler ve belirtilen test veri seti üzerinde değerlendirir. Modelin doğruluğu,
    macro F1 skoru, sınıflandırma raporu ve karışıklık matrisi oluşturulur ve kaydedilir. 
"""

# --- 1. Konfigürasyon ve Argümanlar ---
CHECKPOINT_DIR = "../_train_meta_001"
TEST_DATA_PATH = "../data/data_v2_test_one_hot.csv"

# Model seçimi için Argparse
parser = argparse.ArgumentParser(description="Meta-Model Değerlendirme Betiği")
parser.add_argument(
    '--model_name',
    type=str,
    choices=['lr', 'xgb'],
    default='lr',
    help="Değerlendirilecek model: 'lr' (Logistic Regression) veya 'xgb' (XGBoost)"
)
args = parser.parse_args()
MODEL_NAME = args.model_name
print(f"Değerlendirilecek Model Türü: {MODEL_NAME}")


# --- 2. Gerekli Dosyaları Dinamik Olarak Yükle ---
print(f"\n>>> Adım 1: {MODEL_NAME} modeli, encoder ve test verisi yükleniyor...")

MODEL_DIR = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
MODEL_PATH = os.path.join(MODEL_DIR, f'best_{MODEL_NAME}_meta_model.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, f'best_{MODEL_NAME}_label_encoder.joblib')
SCALER_PATH = os.path.join(CHECKPOINT_DIR, 'lr', 'best_lr_scaler.joblib') # Scaler 'lr' klasöründe

report_filename = f'test_report_full.txt'
cm_filename = 'test_confusion_matrix_full.png'

OUTPUT_REPORT_PATH = os.path.join(MODEL_DIR, report_filename)
OUTPUT_CM_PATH = os.path.join(MODEL_DIR, cm_filename)

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print("Dosyalar başarıyla yüklendi.")
except FileNotFoundError as e:
    print(f"\nHATA: Gerekli bir dosya bulunamadı: {e.filename}")
    print(f"Lütfen önce '{MODEL_NAME}' modeli için eğitim betiğini çalıştırdığınızdan emin olun.")
    exit()

# Scaler'ı SADECE 'lr' modeli için yükle
scaler = None
if MODEL_NAME == 'lr':
    try:
        scaler = joblib.load(SCALER_PATH)
        print("Scaler (lr için gerekli) yüklendi.")
    except FileNotFoundError:
        print(f"HATA: {MODEL_NAME} için 'scaler' dosyası bulunamadı: {SCALER_PATH}")
        exit()

# --- 4. Test Verisini Hazırla ---
print("\n>>> Adım 2: Test verisi özellik (X) ve hedef (y) olarak ayrılıyor...")
y_test = test_df['true_label']
X_test = test_df.drop(columns=['citation_id', 'true_label'])

# Veriyi SADECE 'lr' modeli için ölçeklendir
if scaler is not None:
    print("Veri, yüklenen 'scaler' ile ölçeklendiriliyor (StandardScaler)...")
    X_test_processed = scaler.transform(X_test)
else:
    print("Bu model ('xgb') için ölçeklendirme (scaler) adımı atlanıyor.")
    X_test_processed = X_test

# --- 5. Model ile Tahmin Yap ---
print("\n>>> Adım 3: Yüklenen model ile test seti üzerinde tahminler yapılıyor...")
# Modelin predict metoduna işlenmiş veriyi (X_test_processed) ver
y_pred_encoded = model.predict(X_test_processed)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# --- 6. Performans Raporlarını Yazdır ve Kaydet ---
print("\n>>> Adım 4: Model performansı test seti üzerinde değerlendiriliyor...")
accuracy = accuracy_score(y_test, y_pred_labels)
macro_f1 = f1_score(y_test, y_pred_labels, average='macro')
class_report_str = classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_)

print(f"\nModel Doğruluğu (Accuracy): {accuracy:.4f}")
print(f"Macro F1 Skoru:             {macro_f1:.4f}")
print("\nSınıflandırma Raporu:")
print(class_report_str)

# Raporu dosyaya kaydet
print(f"\nSınıflandırma raporu '{OUTPUT_REPORT_PATH}' dosyasına kaydediliyor...")
with open(OUTPUT_REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Test Verisi: {TEST_DATA_PATH}\n")
    f.write("="*50 + "\n")
    f.write(f"Model Doğruluğu (Accuracy): {accuracy:.4f}\n")
    f.write(f"Macro F1 Skoru:             {macro_f1:.4f}\n")
    f.write("\nSınıflandırma Raporu:\n")
    f.write(class_report_str)

# --- 7. Karışıklık Matrisini (Confusion Matrix) Oluştur ve Kaydet ---
print("\n>>> Adım 5: Karışıklık matrisi oluşturuluyor...")
cm = confusion_matrix(y_test, y_pred_labels, labels=label_encoder.classes_)

# Normalize etme
valid_rows = cm.sum(axis=1) > 0
cm_normalized = np.zeros_like(cm, dtype=float)
cm_normalized[valid_rows] = cm[valid_rows].astype('float') / cm[valid_rows].sum(axis=1)[:, np.newaxis]

# Görselleştirme
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            cbar_kws={'format': '%.2f'})
plt.title(f'Confusion Matrix - {MODEL_NAME.upper()} Model (Test Set)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Grafiği kaydet
plt.savefig(OUTPUT_CM_PATH)

print(f"\nNormalize edilmiş karışıklık matrisi '{OUTPUT_CM_PATH}' olarak başarıyla kaydedildi.")
print("\nDeğerlendirme süreci tamamlandı!")