import pandas as pd
from sqlalchemy import create_engine
import sys

# --- Ayarlar ---

# 1. Veri Kaynakları
SOURCE_CSV_PATH = 'data/data_v2_org.csv'
OUTPUT_CSV_PATH = 'util_all_models_one_hot.csv'
INVALID_LOG_PATH = 'util_invalid_label_log.csv'

# 2. Veritabanı Bağlantısı
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"

# 3. Etiket Bilgileri
LABELS = ['background', 'basis', 'support', 'differ', 'discuss']

# 4. MODELLER
MODELS_TO_PROCESS = [
    {
        'name': 'gemini-flash-k0',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34h1'
    },
    {
        'name': 'gemini-flash-k1',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34a1'
    },
    {
        'name': 'gemini-flash-k2',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34a2'
    },
    {
        'name': 'gemini-flash-k5',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34a5'
    },
    {
        'name': 'chatgpt-4o-k0',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34c1'
    },
    {
        'name': 'chatgpt-4o-k1',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34a6'
    },
    {
        'name': 'chatgpt-4o-k2',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34a7'
    },
    {
        'name': 'chatgpt-4o-k5',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d3410'
    },
    {
        'name': 'dspy',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d34d1'
    },
    {
        'name': 'gemini-pro-k0',
        'user_id': '48ed0fcf-4e78-4913-b96b-d942646d3420'
    },
]


# --- Yardımcı Fonksiyonlar ---

def fetch_source_data(filepath):
    """
    Ana CSV dosyasından ID'leri ve gerçek etiketleri okur.
    """
    try:
        df = pd.read_csv(filepath, usecols=['id', 'citation_intent'])
        df = df.rename(columns={'id': 'citation_id', 'citation_intent': 'true_label'})
        print(f"✅ Kaynak CSV okundu: {filepath} ({len(df)} satır)")
        return df
    except FileNotFoundError:
        print(f"❌ HATA: Kaynak CSV dosyası bulunamadı: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ HATA: Kaynak CSV okunurken bir hata oluştu: {e}")
        sys.exit(1)


def fetch_model_predictions(db_url, user_id, model_name):
    """
    Veritabanından belirli bir modelin (user_id) tüm tahminlerini çeker.
    """
    query = f"""
        SELECT 
            citation_id, 
            citation_intent AS '{model_name}_prediction'
        FROM 
            cec_citation_intent
        WHERE 
            user_id = '{user_id}'
    """
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            df = pd.read_sql(query, connection)
            df = df.drop_duplicates(subset=['citation_id'])
            print(f"✅ Model tahminleri veritabanından çekildi: {model_name} ({len(df)} tahmin)")
            return df
    except Exception as e:
        print(f"❌ HATA: Veritabanı bağlantısı veya sorgu hatası ({model_name}): {e}")
        return pd.DataFrame(columns=['citation_id', f'{model_name}_prediction'])


# --- GÜNCELLENEN FONKSİYON (1. Değişiklik) ---
def create_one_hot_vectors(df, labels, model_name):
    """
    DataFrame'e model tahmini için one-hot sütunları ekler.
    Ayrıca geçerli etiket setinde olmayan tahminleri loglar.
    İki değer döndürür: (güncellenmiş_df, geçersiz_etiketler_df)
    """
    prediction_col = f'{model_name}_prediction'

    invalid_entries_df = pd.DataFrame()

    if prediction_col not in df.columns:
        print(f"⚠️ Uyarı: {model_name} için tahmin sütunu bulunamadı, one-hot ve loglama atlanıyor.")
        return df, invalid_entries_df

    # 1. Geçersiz Etiket Loglaması
    df[prediction_col] = df[prediction_col].fillna('nan_value')
    valid_values = labels + ['nan_value']
    invalid_mask = ~df[prediction_col].isin(valid_values)

    if invalid_mask.any():
        # --- DEĞİŞİKLİK BURADA ---
        # 'true_label' sütununu da log DataFrame'ine ekliyoruz.
        invalid_df = df.loc[invalid_mask, ['citation_id', 'true_label']].copy()
        # --- BİTTİ ---

        invalid_df['model_name'] = model_name
        invalid_df['invalid_prediction'] = df.loc[invalid_mask, prediction_col]

        print(f"⚠️ Bulundu: {len(invalid_df)} geçersiz etiket ({model_name}). Loglanacak.")
        invalid_entries_df = invalid_df

    # 2. One-Hot Encoding
    df[prediction_col] = df[prediction_col].astype(str)

    for label in labels:
        one_hot_col_name = f'{model_name}_{label}'
        df[one_hot_col_name] = (df[prediction_col] == label).astype(int)
    print(f"✅ One-hot vektör sütunları oluşturuldu: {model_name}")

    return df, invalid_entries_df


# --- GÜNCELLENEN Ana Fonksiyon (2. Değişiklik) ---
def main():
    """
    Ana script fonksiyonu - Çoklu model işleme ve hata loglama
    """
    # 1. Adım: Kaynak veriyi oku
    df_main = fetch_source_data(SOURCE_CSV_PATH)

    final_columns = ['citation_id', 'true_label']
    all_invalid_entries = []

    # 2. Adım: Model döngüsü
    for model_info in MODELS_TO_PROCESS:
        model_name = model_info['name']
        user_id = model_info['user_id']

        print(f"\n--- İşleniyor: {model_name} (ID: {user_id}) ---")

        # 3. Adım: Tahminleri çek
        df_model = fetch_model_predictions(DB_URL, user_id, model_name)

        # 4. Adım: Birleştir
        df_main = pd.merge(df_main, df_model, on='citation_id', how='left')

        # 5. Adım: One-hot oluştur VE geçersiz etiketleri al
        df_main, invalid_entries = create_one_hot_vectors(df_main, LABELS, model_name)

        if not invalid_entries.empty:
            all_invalid_entries.append(invalid_entries)

        # 6. Adım: Final sütun adlarını listeye ekle
        for label in LABELS:
            final_columns.append(f'{model_name}_{label}')

    # 7. Adım: Ana One-Hot Çıktısını Kaydet
    df_output = df_main[final_columns]
    df_output.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n🎉 İşlem tamamlandı! Tüm modellerin sonuçları şuraya kaydedildi: {OUTPUT_CSV_PATH}")

    # 8. Adım: Geçersiz Etiket Log'unu Kaydet
    if all_invalid_entries:
        df_invalid_log = pd.concat(all_invalid_entries, ignore_index=True)

        # --- DEĞİŞİKLİK BURADA ---
        # Sütun sırasına 'true_label'ı ekliyoruz.
        log_columns = ['citation_id', 'true_label', 'model_name', 'invalid_prediction']
        # --- BİTTİ ---

        df_invalid_log = df_invalid_log[log_columns]

        df_invalid_log.to_csv(INVALID_LOG_PATH, index=False)
        print(f"ℹ️ {len(df_invalid_log)} adet geçersiz etiket log'u şuraya kaydedildi: {INVALID_LOG_PATH}")
    else:
        print("ℹ️ Hiçbir modelde geçersiz etiket bulunmadı.")


if __name__ == "__main__":
    main()