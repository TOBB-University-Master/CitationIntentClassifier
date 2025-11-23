import pandas as pd
from sqlalchemy import create_engine
import sys

# --- Ayarlar ---

# 1. Veri Kaynakları
SOURCE_CSV_PATH = 'data/data_v2_test.csv'
# Çıktı dosyasının adını daha genel bir hale getirdim:
OUTPUT_CSV_PATH = 'data/data_v2_test_one_hot.csv'

# 2. Veritabanı Bağlantısı
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"

# 3. Etiket Bilgileri
# One-hot vektörü oluşturulacak etiketlerin sırası
LABELS = ['background', 'basis', 'support', 'differ', 'discuss']

# 4. MODELLER (YENİ GÜNCELLENEN BÖLÜM)
# İşlenecek modelleri bir liste olarak buraya ekleyin.
# Her model bir sözlük (dictionary) olmalı.
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
    # model_name'i sütun adı olarak kullanmak, birleştirmede çakışmayı önler
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
            # Tahmini olmayan ID'leri de işleyebilmek için
            # (nadiren de olsa) mükerrer ID'leri temizle
            df = df.drop_duplicates(subset=['citation_id'])
            print(f"✅ Model tahminleri veritabanından çekildi: {model_name} ({len(df)} tahmin)")
            return df
    except Exception as e:
        print(f"❌ HATA: Veritabanı bağlantısı veya sorgu hatası ({model_name}): {e}")
        # Bu model başarısız olursa boş bir DataFrame döndür ki script durmasın
        return pd.DataFrame(columns=['citation_id', f'{model_name}_prediction'])


def create_one_hot_vectors(df, labels, model_name):
    """
    DataFrame'e model tahmini için one-hot sütunları ekler.
    """
    prediction_col = f'{model_name}_prediction'

    # Modelin tahmini yoksa (NaN) hatayı önlemek için str'ye çevir
    # Bu, .astype(int) adımının çalışmasını sağlar
    if prediction_col in df.columns:
        df[prediction_col] = df[prediction_col].astype(str)

        for label in labels:
            one_hot_col_name = f'{model_name}_{label}'
            df[one_hot_col_name] = (df[prediction_col] == label).astype(int)
        print(f"✅ One-hot vektör sütunları oluşturuldu: {model_name}")
    else:
        print(f"⚠️ Uyarı: {model_name} için tahmin sütunu bulunamadı, one-hot oluşturma atlanıyor.")

    return df


# --- Ana Fonksiyon (GÜNCELLENDİ) ---

def main():
    """
    Ana script fonksiyonu - Çoklu model işleme
    """
    # 1. Adım: Kaynak veriyi (tüm ID'ler ve gerçek etiketler) oku
    df_main = fetch_source_data(SOURCE_CSV_PATH)

    # Son CSV'de olmasını istediğimiz sütunların listesini dinamik olarak oluşturalım
    final_columns = ['citation_id', 'true_label']

    # 2. Adım: Her bir model için listeyi döngüye al
    for model_info in MODELS_TO_PROCESS:
        model_name = model_info['name']
        user_id = model_info['user_id']

        print(f"\n--- İşleniyor: {model_name} (ID: {user_id}) ---")

        # 3. Adım: Modelin tahminlerini veritabanından çek
        df_model = fetch_model_predictions(DB_URL, user_id, model_name)

        # 4. Adım: Tahminleri ana DataFrame ile 'citation_id' üzerinden birleştir
        # how='left' kullanarak ana CSV'deki tüm ID'lerin korunmasını sağlarız.
        df_main = pd.merge(df_main, df_model, on='citation_id', how='left')

        # 5. Adım: Bu model için one-hot vektörlerini oluştur
        df_main = create_one_hot_vectors(df_main, LABELS, model_name)

        # 6. Adım: Oluşturulan one-hot sütun adlarını final listeye ekle
        for label in LABELS:
            final_columns.append(f'{model_name}_{label}')

    # 7. Adım: Son çıktıyı kaydet
    # Sadece 'final_columns' listesinde belirttiğimiz sütunları seçer
    # Bu, ara '{model_name}_prediction' sütunlarını otomatik olarak dışarıda bırakır
    df_output = df_main[final_columns]

    df_output.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n🎉 İşlem tamamlandı! Tüm modellerin sonuçları şuraya kaydedildi: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()