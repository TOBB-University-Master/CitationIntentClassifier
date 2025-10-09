import logging
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from thefuzz import fuzz
from tqdm import tqdm
import nltk
import re

# --- NLTK Modellerini Kontrol Et ve İndir ---
def download_nltk_resources():
    """İhtiyaç duyulan NLTK kaynaklarını kontrol eder ve eksikse indirir."""
    required_resources = ['punkt', 'punkt_tab']
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"NLTK '{resource}' modeli zaten mevcut.")
        except LookupError:
            print(f"NLTK '{resource}' modeli bulunamadı. İndiriliyor...")
            nltk.download(resource)
            print(f"'{resource}' indirme tamamlandı.")

# NLTK'nın 'punkt' modelinin indirilmiş olduğundan emin olalım
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' modeli bulunamadı. İndiriliyor...")
    nltk.download('punkt')
    print("İndirme tamamlandı.")

# --- Yapılandırma ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
LOG_FILE = "map_tei_to_citations.log"
NEW_COLUMN_NAME = "citation_context_raw"


# --- Loglama Kurulumu ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )


def add_column_if_not_exists(engine, column_name):
    """
    cec_citation tablosunda belirtilen sütunun varlığını kontrol eder, yoksa oluşturur.
    """
    try:
        with engine.connect() as connection:
            query = text(f"""
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = DATABASE()
                AND table_name = 'cec_citation' AND column_name = '{column_name}'
            """)
            result = connection.execute(query).fetchone()
            if result is None:
                logging.info(f"'{column_name}' sütunu bulunamadı, oluşturuluyor...")
                alter_query = text(f"ALTER TABLE cec_citation ADD COLUMN {column_name} TEXT")
                connection.execute(alter_query)
                if connection.dialect.name == 'mysql': connection.commit()
                logging.info("Sütun başarıyla oluşturuldu.")
            else:
                logging.info(f"'{column_name}' sütunu zaten mevcut.")
    except Exception as e:
        logging.error(f"Sütun kontrolü/oluşturulması sırasında hata: {e}")
        raise


def clean_text_for_matching(s):
    """Karşılaştırma için metinleri temizler."""
    s = re.sub(r'\s+', ' ', s)  # Birden fazla boşluğu tek boşluğa indirge
    s = re.sub(r'\[\d+\]', '', s)  # "[8]" gibi referans numaralarını kaldır
    return s.strip().lower()


def main():
    download_nltk_resources()
    setup_logging()
    logging.info("TEI'den Atıf Eşleştirme Betiği Başlatıldı.")

    engine = create_engine(DB_URL)

    add_column_if_not_exists(engine, NEW_COLUMN_NAME)

    # 1. Adım: Tüm makaleleri ve atıfları çek
    logging.info("Tüm makaleler ve atıflar veritabanından çekiliyor...")
    try:
        articles_df = pd.read_sql("SELECT id, tei_content FROM cec_article_raw WHERE tei_content IS NOT NULL",
                                  engine)
        citations_df = pd.read_sql(
            f"SELECT id, article_id, citation_context FROM cec_citation WHERE citation_context IS NOT NULL AND {NEW_COLUMN_NAME} IS NULL",
            engine)
        if citations_df.empty:
            logging.info("İşlenecek yeni atıf bulunamadı.")
            return
        logging.info(f"Toplam {len(articles_df)} makale ve {len(citations_df)} işlenmemiş atıf bulundu.")
    except Exception as e:
        logging.error(f"Veri çekilirken hata oluştu: {e}")
        return

    updates_to_commit = []

    # 2. Adım: Her bir makale için TEI içeriğini işle
    for _, article in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Makaleler Taranıyor"):
        article_id = article['id']
        tei_content = article['tei_content']

        # O makaleye ait işlenmemiş atıfları al (arama havuzu)
        article_citations = citations_df[citations_df['article_id'] == article_id]
        if article_citations.empty:
            continue

        try:
            soup = BeautifulSoup(tei_content, 'lxml-xml')

            # 3. Adım: TEI içindeki tüm referansları bul
            references = soup.find_all('ref', {'type': 'bibr'})

            for ref in references:
                parent_paragraph = ref.find_parent('p')
                if not parent_paragraph:
                    continue

                # 4. Adım: Referansı içeren cümleyi bul
                # Önce paragrafın orijinal (etiketli) halini ve temiz halini alalım
                p_html = str(parent_paragraph)
                p_text = parent_paragraph.get_text(separator=' ', strip=True)

                # Paragrafı NLTK ile cümlelere ayır
                sentences_in_p = nltk.sent_tokenize(p_text)

                ref_text_clean = ref.get_text(strip=True)

                found_sentence_clean = None
                for sentence in sentences_in_p:
                    if ref_text_clean in sentence:
                        found_sentence_clean = sentence
                        break

                if not found_sentence_clean:
                    continue

                # 5. Adım: Bulunan cümleyi veritabanındaki atıflarla eşleştir
                best_match_id = None
                best_score = -1

                cleaned_tei_sentence = clean_text_for_matching(found_sentence_clean)

                for _, citation in article_citations.iterrows():
                    cleaned_db_context = clean_text_for_matching(citation['citation_context'])
                    score = fuzz.ratio(cleaned_tei_sentence, cleaned_db_context)

                    if score > best_score:
                        best_score = score
                        best_match_id = citation['id']

                # 6. Adım: Yüksek skorlu eşleşme varsa güncelleme listesine ekle
                if best_score > 85:  # Eşik değeri önemli, yüksek tutmakta fayda var
                    # Not: Orijinal cümleyi etiketleriyle birlikte saklamak için özel bir işlem gerekir.
                    # Basit bir çözüm: ref'i ve etrafındaki metni almak.
                    # Ancak şimdilik bulunan temiz cümleyi kaydedelim, bu da değerli bir bilgi.
                    # Daha gelişmiş bir yöntem, orijinal HTML'den cümleyi çıkarmak olurdu.
                    # Şimdilik en pratik olan, bulunan temiz cümleyi kaydetmek.
                    parent_paragraph_with_tags = str(parent_paragraph)
                    updates_to_commit.append({
                        "id": best_match_id,
                        "context_tei": parent_paragraph_with_tags
                    })
                    # Eşleşen atıfı tekrar kullanılmaması için listeden çıkar
                    citations_df = citations_df.drop(citations_df[citations_df.id == best_match_id].index)


        except Exception as e:
            logging.warning(f"Article ID {article_id} işlenirken bir hata oluştu: {e}")

    # 7. Adım: Veritabanını Güncelle
    if updates_to_commit:
        logging.info(f"Toplam {len(updates_to_commit)} kayıt veritabanında güncellenecek...")
        try:
            with engine.connect() as connection:
                trans = connection.begin()
                for update in tqdm(updates_to_commit, desc="Veritabanı Güncelleniyor"):
                    stmt = text(f"UPDATE cec_citation SET {NEW_COLUMN_NAME} = :context_tei WHERE id = :id")
                    connection.execute(stmt, {"context_tei": update["context_tei"], "id": update["id"]})
                trans.commit()
            logging.info("Veritabanı başarıyla güncellendi.")
        except Exception as e:
            logging.error(f"Veritabanı güncellenirken bir hata oluştu: {e}")
            trans.rollback()
    else:
        logging.info("Veritabanını güncellemek için hiçbir eşleşme bulunamadı.")


if __name__ == "__main__":
    main()