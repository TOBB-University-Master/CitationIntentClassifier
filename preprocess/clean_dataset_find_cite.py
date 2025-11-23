from bs4 import BeautifulSoup, NavigableString
from thefuzz import fuzz
from tqdm import tqdm
import re
import logging
import pandas as pd
from sqlalchemy import create_engine, text

# --- Yapılandırma (Değişiklik yok) ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
LOG_FILE = "post_processing_v4.5.log"
SOURCE_COLUMN = "citation_context_raw"
TARGET_COLUMN = "citation_context_ext"


# --- Yardımcı Fonksiyonlar (İlk ikisi değişmedi) ---
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])


def add_column_if_not_exists(engine, column_name):
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
                alter_query = text(
                    f"ALTER TABLE cec_citation ADD COLUMN {column_name} TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                connection.execute(alter_query)
                if connection.dialect.name == 'mysql': connection.commit()
                logging.info("Sütun başarıyla oluşturuldu.")
            else:
                logging.info(f"'{column_name}' sütunu zaten mevcut.")
    except Exception as e:
        logging.error(f"Sütun kontrolü/oluşturulması sırasında hata: {e}")
        raise


def clean_text_for_matching(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    # Metin temizleme işleminde referansları ([13] gibi) korumak skoru artırabilir
    s = re.sub(r'[^a-z0-9\s\u00C0-\u017F\[\]]', '', s)  # Köşeli parantezleri koru
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


# --- YENİ YARDIMCI FONKSİYON ---
def get_context_around_placeholder(plain_text: str, placeholder: str, num_words: int = 15) -> str:
    """
    Bir metin içinde belirli bir placeholder'ın etrafından belirtilen sayıda kelime alır.
    """
    try:
        parts = plain_text.split(placeholder)
        # Placeholder'dan önceki metni kelimelere ayır ve son 'num_words' tanesini al
        words_before = parts[0].strip().split()
        left_context = words_before[-num_words:]

        # Placeholder'dan sonraki metni kelimelere ayır ve ilk 'num_words' tanesini al
        words_after = parts[1].strip().split()
        right_context = words_after[:num_words]

        # Ortasına placeholder'ı da ekleyerek tam bağlamı oluştur
        full_context = " ".join(left_context) + f" {placeholder} " + " ".join(right_context)
        return full_context.strip()
    except Exception:
        # Hata durumunda (örn. placeholder bulunamazsa) tüm metni döndür
        return plain_text


# --- ANA İŞLEM FONKSİYONU (YENİ MANTIKLA GÜNCELLENDİ) ---
def main():
    setup_logging()
    logging.info("Son İşleme Script'i (v4.5 - Bağlam Penceresi Mantığı) Başlatıldı.")
    engine = create_engine(DB_URL)
    add_column_if_not_exists(engine, TARGET_COLUMN)

    try:
        query = f"SELECT id, citation_context, {SOURCE_COLUMN} FROM cec_citation WHERE {SOURCE_COLUMN} IS NOT NULL AND ({TARGET_COLUMN} IS NULL OR {TARGET_COLUMN} = '')"
        df = pd.read_sql(query, engine)
        if df.empty:
            logging.info("İşlenecek yeni veri bulunamadı.")
            return
        logging.info(f"İşlenmek üzere {len(df)} kayıt bulundu.")
    except Exception as e:
        logging.error(f"Veri çekilirken hata oluştu: {e}")
        return

    updates_to_commit = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Paragraflar İşleniyor"):
        citation_id = row['id']
        citation_context = row['citation_context']
        paragraph_html = row[SOURCE_COLUMN]

        if not all(isinstance(s, str) and s.strip() for s in [paragraph_html, citation_context]):
            continue

        try:
            soup = BeautifulSoup(paragraph_html, 'html.parser')
            cleaned_citation = clean_text_for_matching(citation_context)

            all_bibr_refs = soup.find_all('ref', {'type': 'bibr'})

            if not all_bibr_refs:
                logging.warning(f"ID {citation_id}: Paragrafta hiç <ref type='bibr'> bulunamadı. Atlanıyor.")
                continue

            best_ref = None
            if len(all_bibr_refs) == 1:
                best_ref = all_bibr_refs[0]
            else:
                # 1. Her bir ref için geçici bir metin oluştur ve orijinalini sakla
                temp_soup = BeautifulSoup(paragraph_html, 'html.parser')
                temp_refs = temp_soup.find_all('ref', {'type': 'bibr'})

                ref_map = {}
                for i, ref_tag in enumerate(temp_refs):
                    placeholder = f"__REF_{i}__"
                    # placeholder'ı orijinal etikete (soup'tan gelen) haritala
                    ref_map[placeholder] = all_bibr_refs[i]
                    ref_tag.replace_with(placeholder)

                plain_text_with_placeholders = re.sub(r'\s+', ' ', temp_soup.get_text(separator=' ', strip=True))

                # 2. Her bir ref'in bağlamını hedef cümle ile karşılaştır
                highest_score = -1
                best_ref_placeholder = None

                for placeholder in ref_map:
                    context_window = get_context_around_placeholder(plain_text_with_placeholders, placeholder)
                    cleaned_context = clean_text_for_matching(context_window)

                    # fuzz.ratio, iki string'in genel benzerliği için daha iyi çalışır
                    score = fuzz.ratio(cleaned_citation, cleaned_context)

                    if score > highest_score:
                        highest_score = score
                        best_ref_placeholder = placeholder

                if best_ref_placeholder:
                    best_ref = ref_map[best_ref_placeholder]

            if best_ref is None:
                logging.warning(f"ID {citation_id}: En iyi referans belirlenemedi. Atlanıyor.")
                continue

            # 3. Sonucu oluştur: en iyiyi <CITE>, diğerlerini [REF] yap
            #    Bu işlemi orijinal soup üzerinde yapıyoruz
            for tag in soup.find_all(True):
                if tag == best_ref:
                    tag.replace_with('<CITE>')
                elif tag.name == 'ref' and tag.get('type') == 'bibr':
                    tag.replace_with('[REF]')
                else:
                    tag.unwrap()

            final_text = re.sub(r'\s+', ' ', soup.decode_contents().strip()).replace('&lt;CITE&gt;', '<CITE>')
            updates_to_commit.append({"id": citation_id, "ext_data": final_text})

        except Exception as e:
            logging.warning(f"ID {citation_id} işlenirken kritik hata: {e}")

    # --- Veritabanını Güncelle (Değişiklik yok) ---
    if updates_to_commit:
        logging.info(f"{len(updates_to_commit)} kayıt veritabanında güncellenecek...")
        try:
            with engine.connect() as connection:
                trans = connection.begin()
                for update in tqdm(updates_to_commit, desc="Veritabanı Güncelleniyor"):
                    stmt = text(f"UPDATE cec_citation SET {TARGET_COLUMN} = :ext_data WHERE id = :id")
                    connection.execute(stmt, {"ext_data": update["ext_data"], "id": update["id"]})
                trans.commit()
            logging.info("Veritabanı başarıyla güncellendi.")
        except Exception as e:
            logging.error(f"Veritabanı güncellenirken bir hata oluştu: {e}")
            if 'trans' in locals() and trans.is_active:
                trans.rollback()


if __name__ == "__main__":
    main()