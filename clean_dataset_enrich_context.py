import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup
from thefuzz import fuzz
from tqdm import tqdm

# --- Yapılandırma ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
LOG_FILE = "enrich_context.log"


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


def add_column_if_not_exists(engine):
    """
    cec_citation tablosunda citation_context_ext sütununun varlığını kontrol eder,
    yoksa oluşturur.
    """
    try:
        with engine.connect() as connection:
            # Sütunun varlığını kontrol etmek için bir sorgu
            # Bu sorgu veritabanı sistemine göre değişebilir, bu MySQL/MariaDB için.
            query = text("""
                         SELECT 1
                         FROM information_schema.columns
                         WHERE table_schema = DATABASE()
                           AND table_name = 'cec_citation'
                           AND column_name = 'citation_context_ext'
                         """)
            result = connection.execute(query).fetchone()

            if result is None:
                logging.info("'citation_context_ext' sütunu bulunamadı, oluşturuluyor...")
                alter_query = text("ALTER TABLE cec_citation ADD COLUMN citation_context_ext TEXT")
                connection.execute(alter_query)
                # MySQL'de alter table'dan sonra commit gerekir.
                if connection.dialect.name == 'mysql':
                    connection.commit()
                logging.info("Sütun başarıyla oluşturuldu.")
            else:
                logging.info("'citation_context_ext' sütunu zaten mevcut.")
    except Exception as e:
        logging.error(f"Sütun kontrolü/oluşturulması sırasında hata: {e}")
        raise


def main():
    setup_logging()
    logging.info("Bağlam Zenginleştirme Betiği Başlatıldı.")

    engine = create_engine(DB_URL)

    # 1. Column yoksa eklenir
    add_column_if_not_exists(engine)

    # 2. İşlenmemiş atıfları veritabanından çek
    logging.info("İşlenmemiş atıflar veritabanından çekiliyor (citation_context_ext IS NULL)...")
    try:
        citations_df = pd.read_sql(
            "SELECT id, citation_context, article_id FROM cec_citation WHERE citation_context_ext IS NULL AND citation_context IS NOT NULL",
            engine
        )
        if citations_df.empty:
            logging.info("İşlenecek yeni atıf bulunamadı. İşlem tamamlandı.")
            return
        logging.info(f"Toplam {len(citations_df)} adet işlenmemiş atıf bulundu.")
    except Exception as e:
        logging.error(f"Atıflar çekilirken hata oluştu: {e}")
        return

    updates_to_commit = []

    # 3. Atıfları makale bazında gruplayarak işle
    grouped_citations = citations_df.groupby('article_id')
    for article_id, group in tqdm(grouped_citations, desc="Makaleler İşleniyor"):
        try:
            # 4. İlgili makalenin TEI içeriğini çek
            tei_content_result = pd.read_sql(
                "SELECT tei_content FROM cec_article_raw WHERE id = %(article_id)s",
                engine,
                params={"article_id": article_id}
            )

            if tei_content_result.empty or tei_content_result.iloc[0]['tei_content'] is None:
                logging.warning(f"Article ID {article_id} için TEI içeriği bulunamadı, atlanıyor.")
                continue

            tei_content = tei_content_result.iloc[0]['tei_content']

            # 5. TEI içeriğini BeautifulSoup ile ayrıştır ve paragrafları bul
            soup = BeautifulSoup(tei_content, 'lxml-xml')  # 'lxml-xml' parser'ını kullanmak daha hızlı ve güvenilir
            paragraphs = soup.find_all('p')

            if not paragraphs:
                logging.warning(f"Article ID {article_id} için TEI içinde <p> etiketi bulunamadı.")
                continue

            # Paragrafların temiz metinlerini bir listeye alıyoruz
            paragraph_texts = [p.get_text(separator=' ', strip=True) for p in paragraphs]

            # 6. Bu makaledeki her bir atıf için en iyi paragrafı bul
            for index, citation in group.iterrows():
                citation_id = citation['id']
                citation_context = citation['citation_context'].strip()

                best_match_paragraph = None
                best_score = -1

                for p_text in paragraph_texts:
                    # thefuzz kütüphanesi ile benzerlik oranını hesapla
                    score = fuzz.partial_ratio(citation_context, p_text)

                    if score > best_score:
                        best_score = score
                        best_match_paragraph = p_text

                # Eşiğin üzerindeyse güncelleme listesine eklenir
                if best_score >= 65:
                    updates_to_commit.append({
                        "id": citation_id,
                        "context_ext": best_match_paragraph
                    })
                else:
                    logging.warning(
                        f"Citation ID {citation_id} için yeterli benzerlikte paragraf bulunamadı (En Yüksek Skor: {best_score}).")

        except Exception as e:
            logging.error(f"Article ID {article_id} işlenirken bir hata oluştu: {e}")
            continue

    # 7. Toplanan güncellemeleri veritabanına yaz
    if updates_to_commit:
        logging.info(f"Toplam {len(updates_to_commit)} kayıt veritabanında güncellenecek...")
        try:
            with engine.connect() as connection:
                trans = connection.begin()
                for update in tqdm(updates_to_commit, desc="Veritabanı Güncelleniyor"):
                    stmt = text("""
                                UPDATE cec_citation
                                SET citation_context_ext = :context_ext
                                WHERE id = :id
                                """)
                    connection.execute(stmt, {"context_ext": update["context_ext"], "id": update["id"]})
                trans.commit()
            logging.info("Veritabanı başarıyla güncellendi.")
        except Exception as e:
            logging.error(f"Veritabanı güncellenirken bir hata oluştu: {e}")
            trans.rollback()
    else:
        logging.info("Veritabanını güncellemek için hiçbir kayıt bulunamadı.")


if __name__ == "__main__":
    main()