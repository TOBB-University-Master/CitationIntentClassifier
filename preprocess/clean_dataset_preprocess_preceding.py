import pandas as pd
from nltk.tokenize import sent_tokenize
from sqlalchemy import create_engine, text
import sys

# --- Ayarlar: Bu bölümde bir değişiklik yapmanıza gerek yok ---

DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
TABLE_NAME = "cec_citation"
PRIMARY_KEY_COL = "id"
SOURCE_COL = "citation_context_ext"
TARGET_COL = "citation_context_pre"


# --------------------------------------------------------------------

def extract_citation_context(text_input: str) -> str:
    """
    Verilen metin içindeki cümleleri ayırır, '<CITE>' içeren cümlenin
    bir öncesi ve bir sonrasıyla birlikte birleştirir.
    """
    if not isinstance(text_input, str) or '<CITE>' not in text_input:
        return ""

    try:
        sentences = sent_tokenize(text_input, language='turkish')
    except Exception as e:
        print(f"\n[Hata] NLTK tokenization hatası: {e}\nMetin: {text_input[:100]}...")
        return text_input

    cite_index = -1
    for i, sentence in enumerate(sentences):
        if '<CITE>' in sentence:
            cite_index = i
            break

    if cite_index == -1:
        return " ".join(sentences)

    start_index = max(0, cite_index - 1)
    end_index = min(len(sentences), cite_index + 2)

    context_sentences = sentences[start_index:end_index]
    return " ".join(context_sentences)


def process_database_with_url():
    """
    DB_URL kullanarak veritabanına bağlanır, veriyi işler ve sonuçları günceller.
    """
    try:
        engine = create_engine(DB_URL)
        print("Veritabanı motoru başarıyla oluşturuldu.")

        # === DÜZELTİLEN BÖLÜM ===
        # Bağlantıyı ve transaction'ı tek bir 'with' bloğunda yönetiyoruz.
        with engine.begin() as connection:
            print("Veritabanına bağlanıldı ve transaction başlatıldı.")

            # 1. Veriyi Oku
            query = f"SELECT {PRIMARY_KEY_COL}, {SOURCE_COL} FROM {TABLE_NAME}"
            print(f"Veri okunuyor: {query}")
            df = pd.read_sql_query(sql=text(query), con=connection)
            print(f"{len(df)} satır veri okundu.")

            if df.empty:
                print("Tabloda işlenecek veri bulunamadı.")
                return

            # 2. Veriyi İşle
            print("Cümle bağlamları ayıklanıyor...")
            df[SOURCE_COL] = df[SOURCE_COL].fillna('')
            df[TARGET_COL] = df[SOURCE_COL].apply(extract_citation_context)
            print("İşlem tamamlandı.")

            # 3. Veritabanını Güncelle
            print("Veritabanı satırları güncelleniyor...")
            update_query = text(f"UPDATE {TABLE_NAME} SET {TARGET_COL} = :target WHERE {PRIMARY_KEY_COL} = :pkey")

            update_count = 0
            for index, row in df.iterrows():
                connection.execute(update_query, {"target": row[TARGET_COL], "pkey": row[PRIMARY_KEY_COL]})
                update_count += 1

                if (update_count % 100 == 0) or (update_count == len(df)):
                    sys.stdout.write(f"\r{update_count}/{len(df)} satır güncellendi.")
                    sys.stdout.flush()

            # Bu bloktan hatasız çıkıldığında transaction otomatik olarak commit edilir.
            print(f"\nGüncelleme tamamlandı. Değişiklikler veritabanına işlendi (commit).")

    # Herhangi bir hata oluşursa 'with engine.begin()' bloğu otomatik olarak rollback yapar.
    except Exception as e:
        print(f"\nİşlem sırasında bir hata oluştu: {e}")
        print("Değişiklikler otomatik olarak geri alındı (rollback). Lütfen hatayı kontrol edin.")


# Ana fonksiyonu çalıştır
if __name__ == "__main__":
    process_database_with_url()