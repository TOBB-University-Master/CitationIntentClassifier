import logging
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from thefuzz import fuzz
from tqdm import tqdm
import re
import json

# --- Yapılandırma ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
LOG_FILE = "clean_dataset_find_paragraph.log"
NEW_COLUMN_NAME = "citation_context_raw"
MATCH_THRESHOLD = 55


# --- Yardımcı Fonksiyonlar (Öncekiyle aynı) ---
def setup_logging():
    """Loglama ayarlarını yapar."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler()
        ]
    )


def add_column_if_not_exists(engine, column_name):
    """Belirtilen sütun yoksa veritabanı tablosuna ekler."""
    try:
        with engine.connect() as connection:
            query = text(f"""
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = DATABASE()
                AND table_name = 'cec_citation' AND column_name = '{column_name}'
            """)
            if connection.execute(query).fetchone() is None:
                logging.info(f"'{column_name}' sütunu bulunamadı, oluşturuluyor...")
                alter_query = text(f"ALTER TABLE cec_citation ADD COLUMN {column_name} LONGTEXT")
                connection.execute(alter_query)
                if connection.dialect.name == 'mysql': connection.commit()
                logging.info("Sütun başarıyla oluşturuldu.")
            else:
                logging.info(f"'{column_name}' sütunu zaten mevcut.")
    except Exception as e:
        logging.error(f"Sütun kontrolü/oluşturulması sırasında hata: {e}")
        raise


def clean_text_for_matching(s: str) -> str:
    """Karşılaştırma için metinleri temizler."""
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', s)
    s = re.sub(r'[^a-z0-9\s\u00C0-\u017F]', '', s)  # Türkçe karakterleri koru
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# --- Ana İşlem Fonksiyonu ---
def main():
    setup_logging()
    logging.info("TEI'den Atıf Eşleştirme Betiği (v3.1 - Ham HTML Kaydetme) Başlatıldı.")

    engine = create_engine(DB_URL)
    add_column_if_not_exists(engine, NEW_COLUMN_NAME)

    try:
        logging.info("Tüm makaleler belleğe yükleniyor...")
        articles_df = pd.read_sql("SELECT id, tei_content FROM cec_article_raw WHERE tei_content IS NOT NULL", engine)
        articles_dict = pd.Series(articles_df.tei_content.values, index=articles_df.id).to_dict()
        logging.info(f"{len(articles_dict)} makale yüklendi.")

        citations_df = pd.read_sql(
            f"SELECT id, article_id, citation_context FROM cec_citation WHERE citation_context IS NOT NULL AND {NEW_COLUMN_NAME} IS NULL",
            engine)
        if citations_df.empty:
            logging.info("İşlenecek yeni atıf bulunamadı. Script sonlandırılıyor.")
            return
        logging.info(f"{len(citations_df)} işlenmemiş atıf bulundu.")

    except Exception as e:
        logging.error(f"Veri çekilirken hata oluştu: {e}")
        return

    updates_to_commit = []

    for _, citation in tqdm(citations_df.iterrows(), total=len(citations_df), desc="Atıflar Taranıyor"):
        citation_id = citation['id']
        article_id = citation['article_id']
        citation_context = citation['citation_context']

        tei_content = articles_dict.get(article_id)
        if not tei_content:
            continue

        try:
            soup = BeautifulSoup(tei_content, 'lxml-xml')
            paragraphs = soup.find_all('p')
            if not paragraphs:
                continue

            best_paragraph_html = None
            best_score = -1

            cleaned_citation_context = clean_text_for_matching(citation_context)

            for p_tag in paragraphs:
                p_text = p_tag.get_text(separator=' ', strip=True)
                cleaned_p_text = clean_text_for_matching(p_text)

                if not cleaned_p_text:
                    continue

                score = fuzz.token_set_ratio(cleaned_citation_context, cleaned_p_text)

                if score > best_score:
                    best_score = score
                    best_paragraph_html = str(p_tag)

            if best_score > MATCH_THRESHOLD:
                # DEĞİŞİKLİK BURADA: Artık JSON oluşturmuyoruz.
                updates_to_commit.append({
                    "id": citation_id,
                    "paragraph_html": best_paragraph_html  # Doğrudan paragrafın HTML'ini ekliyoruz.
                })

        except Exception as e:
            logging.warning(f"Atıf ID {citation_id} (Makale ID {article_id}) işlenirken hata oluştu: {e}")

    # --- Veritabanını Güncelle ---
    if updates_to_commit:
        logging.info(f"Toplam {len(updates_to_commit)} kayıt veritabanında güncellenecek...")
        try:
            with engine.connect() as connection:
                trans = connection.begin()
                for update in tqdm(updates_to_commit, desc="Veritabanı Güncelleniyor"):
                    # DEĞİŞİKLİK BURADA: SQL sorgusu ve parametreler basitleştirildi.
                    stmt = text(f"UPDATE cec_citation SET {NEW_COLUMN_NAME} = :paragraph_html WHERE id = :id")
                    connection.execute(stmt, {"paragraph_html": update["paragraph_html"], "id": update["id"]})
                trans.commit()
            logging.info("Veritabanı başarıyla güncellendi.")
        except Exception as e:
            logging.error(f"Veritabanı güncellenirken bir hata oluştu: {e}")
            trans.rollback()
    else:
        logging.info("Veritabanını güncellemek için hiçbir eşleşme bulunamadı.")


if __name__ == "__main__":
    main()