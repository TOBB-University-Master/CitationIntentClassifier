import google.generativeai as genai
import pandas as pd
import os
import time
from tqdm import tqdm

# --- 1. AYARLAR ---
# ;GEMINI_API_KEY=XXXXX
API_KEY = os.environ.get('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmamış. Lütfen API anahtarınızı ayarlayın.")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# Artırılacak azınlık sınıfları
TARGET_CLASSES = ['basis', 'differ', 'discuss', 'support']

# Her bir orijinal örnek için kaç yeni örnek üretilsin?
N_EXAMPLES_PER_PROMPT = 4

# Girdi ve Çıktı Dosya Yolları
INPUT_CSV = "data/data_v2.csv"
OUTPUT_CSV = "data/augmented_data_for_review.csv"

# --- 2. PROMPT (İSTEM) TASLAĞI ---
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

# --- 3.5. DAHA ÖNCE İŞLENENLERİ KONTROL ET --- # <-- YENİ BÖLÜM
processed_contexts_set = set()
df_existing_augmented = pd.DataFrame()  # Önceden üretilmiş verileri tutmak için

try:
    # Mevcut çıktı dosyasını oku
    df_existing_augmented = pd.read_csv(OUTPUT_CSV)

    # 'original_context' sütunu varsa, bu değerleri sete ekle
    if 'original_context' in df_existing_augmented.columns:
        processed_contexts_set = set(df_existing_augmented['original_context'].dropna())
        print(
            f"'{OUTPUT_CSV}' bulundu. {len(processed_contexts_set)} adet daha önce işlenmiş 'original_context' yüklendi.")
    else:
        print(f"'{OUTPUT_CSV}' bulundu ancak 'original_context' sütunu yok. Yeniden oluşturulacak.")
        df_existing_augmented = pd.DataFrame()  # DataFrame'i sıfırla

except (FileNotFoundError, pd.errors.EmptyDataError):
    print(f"'{OUTPUT_CSV}' bulunamadı veya boş. Yeni bir dosya oluşturulacak.")
    # df_existing_augmented zaten boş bir DataFrame olarak tanımlanmıştı

# --- 4. VERİ ARTIRMA İŞLEMİ ---

# Gemini modelini yapılandır
model = genai.GenerativeModel(MODEL_NAME)

# Üretilen YENİ verileri saklamak için bir liste
augmented_data_list = []  # Sadece bu çalıştırmadaki yeni verileri tutar

print(f"'{MODEL_NAME}' modeli ile veri artırma işlemi başlıyor...")
print(f"Toplam {len(df_minority)} kaynak örnek kontrol edilecek...")

# tqdm ile bir ilerleme çubuğu ekleyelim
for index, row in tqdm(df_minority.iterrows(), total=df_minority.shape[0], desc="Örnekler Artırılıyor"):
    original_context = row['citation_context']
    original_intent = row['citation_intent']
    original_section = row['section']

    if original_context in processed_contexts_set:
        continue

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

        time.sleep(1)

    except Exception as e:
        print(f"HATA: '{original_context[:50]}...' örneği işlenirken hata oluştu: {e}")
        continue


df_newly_augmented = pd.DataFrame(augmented_data_list)
df_combined = pd.concat([df_existing_augmented, df_newly_augmented], ignore_index=True)
df_combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

print("\n--- İŞLEM TAMAMLANDI ---")
if not df_newly_augmented.empty:
    print(f"Bu çalıştırmada {len(df_newly_augmented)} adet YENİ sentetik örnek üretildi.")
else:
    print("Bu çalıştırmada yeni örnek üretilmedi (muhtemelen tümü daha önce işlenmiş).")

print(f"Toplam {len(df_combined)} adet sentetik örnek '{OUTPUT_CSV}' dosyasına kaydedildi.")