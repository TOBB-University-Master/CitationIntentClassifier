import google.generativeai as genai
import pandas as pd
import os
import time
from tqdm import tqdm

# --- 1. AYARLAR ---

# API Anahtarınızı ortam değişkeninden alın
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmamış. Lütfen API anahtarınızı ayarlayın.")

genai.configure(api_key=API_KEY)

# Makalede bahsedilen modellerden birini seçelim [cite: 35, 81]
# 'gemini-1.5-flash' daha hızlı ve maliyet etkindir.
MODEL_NAME = "gemini-2.5-flash"

# Artırılacak azınlık sınıfları
TARGET_CLASSES = ['basis', 'differ', 'discuss', 'support']

# Her bir orijinal örnek için kaç yeni örnek üretilsin?
# Makale 3 örnek üretmişti
N_EXAMPLES_PER_PROMPT = 4

# Girdi ve Çıktı Dosya Yolları
INPUT_CSV = "data/data_v2.csv"
OUTPUT_CSV = "data/augmented_data_for_review.csv"

# --- 2. PROMPT (İSTEM) TASLAĞI ---

# Bu istem, LLM'e tam olarak ne yapması gerektiğini söyler.
# <CITE> kuralını ve sınıfı korumasını vurgular.
PROMPT_TEMPLATE = """
Sen, akademik metinler konusunda uzman bir veri artırma asistanısın.
Görevin, bir atıf sınıflandırma görevi için sentetik eğitim verisi üretmek.

Sana bir "Orijinal Atıf Cümlesi" ve bu cümlenin "Atıf Niyeti Sınıfı"nı vereceğim.

Senden beklentilerim:
1.  Orijinal cümlenin anlamsal bütünlüğünü ve verilen "Atıf Niyeti Sınıfı"nı koru.
2.  Cümledeki atıfı belirtmek için mutlaka `<CITE>` özel token'ını kullan.
3.  Orijinal cümleden farklı, ancak aynı niyeti taşıyan {num_examples} adet yeni cümle üret.
4.  Sadece üretilen yeni cümleleri listele, başka hiçbir açıklama veya giriş metni ekleme. Her cümleyi yeni bir satıra yaz.

---
Orijinal Atıf Cümlesi: "{context}"
Atıf Niyeti Sınıfı: "{intent}"
---

{num_examples} adet yeni sentetik cümlen:
"""

# --- 3. VERİ YÜKLEME VE FİLTRELEME ---

try:
    df_original = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"HATA: '{INPUT_CSV}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Sadece artırmak istediğimiz azınlık sınıflarını filtrele
df_minority = df_original[df_original['citation_intent'].isin(TARGET_CLASSES)].copy()

print(f"'{INPUT_CSV}' yüklendi. Toplam {len(df_original)} örnek.")
print(f"Azınlık sınıfları ({', '.join(TARGET_CLASSES)}) için {len(df_minority)} örnek bulundu ve bunlar artırılacak.")

# --- 4. VERİ ARTIRMA İŞLEMİ ---

# Gemini modelini yapılandır
model = genai.GenerativeModel(MODEL_NAME)

# Üretilen yeni verileri saklamak için bir liste
augmented_data_list = []

print(f"'{MODEL_NAME}' modeli ile veri artırma işlemi başlıyor...")

# tqdm ile bir ilerleme çubuğu ekleyelim
for index, row in tqdm(df_minority.iterrows(), total=df_minority.shape[0], desc="Örnekler Artırılıyor"):
    original_context = row['citation_context']
    original_intent = row['citation_intent']
    original_section = row['section']

    # Bu örnek için prompt'u oluştur
    prompt = PROMPT_TEMPLATE.format(
        num_examples=N_EXAMPLES_PER_PROMPT,
        context=original_context,
        intent=original_intent
    )

    try:
        # API'ye isteği gönder
        response = model.generate_content(prompt)

        # Gelen cevabı satırlara böl
        new_contexts = response.text.strip().split('\n')

        # Üretilen her yeni cümleyi listeye ekle
        for new_ctx in new_contexts:
            new_ctx_clean = new_ctx.strip()
            if new_ctx_clean:  # Boş satırları atla
                augmented_data_list.append({
                    "new_context": new_ctx_clean,
                    "original_context": original_context,  # Karşılaştırma için
                    "citation_intent": original_intent,  # Orijinal sınıfı koru
                    "section": "synthetic-augmentation"  # Nereden geldiğini bil
                })

        # API hız limitlerine takılmamak için kısa bir bekleme
        time.sleep(1)  # Saniyede 1 istek (Flash modeli için genellikle yeterli)

    except Exception as e:
        # Bu, konuştuğumuz "Format Kontrolü"nün bir parçasıdır
        # Hata olursa (örn: API hatası, içerik filtresi)
        print(f"HATA: '{original_context[:50]}...' örneği işlenirken hata oluştu: {e}")
        continue

# --- 5. SONUÇLARI KAYDETME ---

# Listeyi DataFrame'e dönüştür
df_augmented = pd.DataFrame(augmented_data_list)

# Manuel inceleme için CSV olarak kaydet
df_augmented.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

print("\n--- İŞLEM TAMAMLANDI ---")
print(f"Toplam {len(df_augmented)} adet yeni sentetik örnek üretildi.")
print(f"Sonuçlar manuel inceleme için '{OUTPUT_CSV}' dosyasına kaydedildi.")