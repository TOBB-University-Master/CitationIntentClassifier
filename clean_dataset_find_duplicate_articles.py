import xml.etree.ElementTree as ET
from collections import defaultdict
import sys
from sqlalchemy import create_engine, text
from typing import Union  # <-- BU SATIRI EKLEYİN

# --- VERİTABANI BAĞLANTI BİLGİLERİ ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"

# TEI XML'in kullandığı standart namespace.
TEI_NAMESPACE = {'tei': 'http://www.tei-c.org/ns/1.0'}


def get_db_engine():
    """SQLAlchemy engine nesnesini oluşturur."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as connection:
            pass
        return engine
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        sys.exit(1)


# Düzeltilmiş fonksiyon imzası
def extract_title_from_tei(tei_content: str) -> Union[str, None]:
    """
    Verilen TEI formatındaki string'den makale başlığını çeker.
    """
    if not tei_content:
        return None
    try:
        root = ET.fromstring(tei_content)
        title_element = root.find('.//tei:titleStmt/tei:title', TEI_NAMESPACE)

        if title_element is not None and title_element.text:
            return title_element.text.strip()
        else:
            return None
    except ET.ParseError:
        return None


def find_duplicate_articles():
    """
    Veritabanından makaleleri çeker ve başlığı aynı olanları bulur.
    """
    engine = get_db_engine()
    titles_map = defaultdict(list)

    print("Veritabanına bağlanılıyor ve kayıtlar okunuyor...")

    try:
        with engine.connect() as connection:
            query = text("SELECT id, tei_content FROM cec_article_raw;")
            result = connection.execute(query)

            rows = result.fetchall()

            print(f"Toplam {len(rows)} kayıt bulundu. Başlıklar ayrıştırılıyor...")

            for row_id, tei_content in rows:
                title = extract_title_from_tei(tei_content)

                if title:
                    titles_map[title].append(row_id)

    except Exception as e:
        print(f"Veritabanı sorgulama hatası: {e}")

    print("\n--- Yinelenen Makale Başlıkları ---")

    found_duplicates = False
    for title, ids in titles_map.items():
        if len(ids) > 1:
            found_duplicates = True
            print(f"\nBaşlık: {title}")
            print(f"  -> Bu başlık {len(ids)} kez bulundu. Kayıt ID'leri: {ids}")

    if not found_duplicates:
        print("Yinelenen başlığa sahip makale bulunamadı.")


if __name__ == "__main__":
    find_duplicate_articles()