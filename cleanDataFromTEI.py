import pandas as pd
from sqlalchemy import create_engine, text
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
import re
from thefuzz import fuzz
import copy
import logging

# --- Yapılandırma ---
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
MATCH_THRESHOLD = 85
LOG_FILE = "update_citations.log"


# --- Loglama Kurulumu ---
def setup_logging():
    # ... (değişiklik yok)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])


# --- Veritabanı Hazırlık Fonksiyonu ---
def prepare_database_columns(engine):
    # ... (değişiklik yok)
    columns_to_check = ["citation_context_raw", "citation_context_ext"]
    try:
        with engine.connect() as connection:
            for column_name in columns_to_check:
                query = text(
                    f"SELECT 1 FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = 'cec_citation' AND column_name = '{column_name}'")
                result = connection.execute(query).fetchone()
                if result is None:
                    logging.info(f"'{column_name}' sütunu bulunamadı, oluşturuluyor...")
                    alter_query = text(f"ALTER TABLE cec_citation ADD COLUMN {column_name} TEXT")
                    connection.execute(alter_query)
                    if connection.dialect.name == 'mysql': connection.commit()
                    logging.info(f"'{column_name}' sütunu başarıyla oluşturuldu.")
                else:
                    logging.info(f"'{column_name}' sütunu zaten mevcut.")
    except Exception as e:
        logging.error(f"Sütun kontrolü/oluşturulması sırasında hata: {e}")
        raise


# --- Diğer Yardımcı Fonksiyonlar ---
def fetch_data_to_process(engine):
    # ... (değişiklik yok)
    logging.info("Veritabanından tüm makaleler ve işlenmemiş atıflar alınıyor...")
    try:
        articles_df = pd.read_sql(
            "SELECT id, tei_content FROM cec_article_raw WHERE tei_content IS NOT NULL AND tei_content != ''", engine)
        citations_df = pd.read_sql(
            "SELECT id, article_id, citation_context FROM cec_citation WHERE citation_context IS NOT NULL AND (citation_context_raw IS NULL OR citation_context_ext IS NULL)",
            engine)
        if articles_df.empty:
            logging.warning("`cec_article_raw` tablosunda işlenecek makale bulunamadı.")
            return None, None
        if citations_df.empty:
            logging.info("İşlenecek yeni atıf bulunamadı. Tüm kayıtlar güncel.")
            return articles_df, None
        logging.info(f"İşlenmek üzere {len(articles_df)} adet makale ve {len(citations_df)} adet atıf bulundu.")
        return articles_df, citations_df
    except Exception as e:
        logging.error(f"Veri çekilirken hata oluştu: {e}")
        return None, None


# --- GÜNCELLENMİŞ FONKSİYON ---
def find_sentences_with_refs(p_tag):
    results = []
    refs_in_p = p_tag.find_all('ref', {'type': 'bibr'})

    sentence_break_pattern = re.compile(r'(?<=[.?!])\s+(?=[A-ZÇĞİÖŞÜ<]|\d)')

    for ref in refs_in_p:
        sentence_parts = [str(ref)]

        # Geriye doğru giderek cümlenin başını bul
        for sibling in ref.previous_siblings:
            if isinstance(sibling, NavigableString):
                text = str(sibling)
                parts = sentence_break_pattern.split(text)
                if len(parts) > 1:
                    sentence_parts.insert(0, parts[-1])
                    break
                else:
                    sentence_parts.insert(0, text)
            else:
                sentence_parts.insert(0, str(sibling))

        # İleriye doğru giderek cümlenin sonunu bul
        for sibling in ref.next_siblings:
            if isinstance(sibling, NavigableString):
                text = str(sibling)
                # İleriye giderken sadece ilk eşleşmeyi ararız
                match = sentence_break_pattern.search(text)
                if match:
                    # Eşleşmenin başlangıcına kadar olan kısmı al
                    first_part = text[:match.start()]
                    sentence_parts.append(first_part)
                    # Cümle sonu noktalama işaretini de ekle
                    end_char_match = re.search(r'([.?!])', first_part)
                    if not end_char_match:  # Eğer böldüğümüz kısımda yoksa, orijinal metinden bul
                        end_char_match = re.search(r'([.?!])', text)
                    if end_char_match:
                        sentence_parts.append(end_char_match.group(1))
                    break  # Döngüyü kır
                else:
                    sentence_parts.append(text)
            else:
                sentence_parts.append(str(sibling))

        full_sentence = "".join(sentence_parts).strip()
        results.append((full_sentence, ref))

    return results


def clean_text_for_matching(s):
    # ... (değişiklik yok)
    if '<' in s and '>' in s:
        s = BeautifulSoup(s, 'lxml').get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', s).strip()


def process_article_and_collect_updates(tei_content, article_id, all_citations_df):
    updates_for_this_article = []
    if not tei_content: return updates_for_this_article
    db_citations = all_citations_df[all_citations_df['article_id'] == article_id]
    if db_citations.empty: return updates_for_this_article

    soup = BeautifulSoup(tei_content, 'lxml-xml')
    paragraphs = soup.find_all('p')

    for p_tag in paragraphs:
        ref_data_list = find_sentences_with_refs(p_tag)
        for tei_sentence, ref_tag_obj in ref_data_list:
            clean_tei_sentence = clean_text_for_matching(tei_sentence)
            best_match_row, best_score = None, -1

            for _, db_row in db_citations.iterrows():
                clean_db_context = clean_text_for_matching(db_row['citation_context'])
                score = fuzz.ratio(clean_tei_sentence, clean_db_context)
                if score > best_score:
                    best_score = score
                    best_match_row = db_row

            # --- DEĞİŞİKLİK BU BLOK İÇİNDE ---
            if best_score >= MATCH_THRESHOLD:
                # 1. Adım: Orijinal paragrafın ham (etiketli) halini al.
                original_p_str = str(p_tag)

                # 2. Adım: modified_p_str'ı oluşturmak için paragrafın kopyası üzerinde çalış.
                p_tag_copy = copy.deepcopy(p_tag)

                # Hedef <ref> etiketinin metin halini al (karşılaştırma için).
                target_ref_str = str(ref_tag_obj)

                # Geçici, benzersiz bir yer tutucu belirle.
                cite_placeholder = "___CITE_PLACEHOLDER___"

                # Kopyalanan paragraftaki tüm <ref>'leri bul.
                all_refs_in_copy = p_tag_copy.find_all('ref')

                # Hedef <ref>'i yer tutucu ile değiştir, diğerlerini yok et.
                for ref in all_refs_in_copy:
                    if str(ref) == target_ref_str:
                        # Eşleşen ref'i yer tutucu ile değiştir.
                        ref.replace_with(cite_placeholder)
                    else:
                        # Diğer tüm ref'leri içerikleriyle birlikte kaldır.
                        ref.decompose()

                # 3. Adım: Artık paragrafın temiz metnini alabiliriz.
                # .get_text() metodu, <p> dahil tüm etiketleri kaldıracaktır.
                modified_p_text = p_tag_copy.get_text(separator=' ', strip=True)

                # 4. Adım: Yer tutucuyu son <CITE> token'ı ile değiştir.
                modified_p_str = modified_p_text.replace(cite_placeholder, "<CITE>")

                # 5. Adım: Sonuçları güncelleme listesine ekle.
                updates_for_this_article.append({
                    "id": best_match_row['id'],
                    "citation_context_raw": original_p_str,
                    "citation_context_ext": modified_p_str
                })
            else:
                print(str(best_score))
                print(str(p_tag))
                print(str(ref_tag_obj))
                print(clean_tei_sentence)
                exit(0)

    return updates_for_this_article

def commit_updates_to_db(engine, updates):
    # ... (değişiklik yok)
    if not updates:
        logging.info("Veritabanını güncellemek için hiçbir kayıt bulunamadı.")
        return

    logging.info(f"Toplam {len(updates)} kayıt veritabanında güncellenecek...")
    try:
        with engine.connect() as connection:
            trans = connection.begin()
            for update in tqdm(updates, desc="Veritabanı Güncelleniyor"):
                stmt = text(""" UPDATE cec_citation
                                SET citation_context_raw = :raw,
                                    citation_context_ext = :ext
                                WHERE id = :id """)
                connection.execute(stmt, {"raw": update["citation_context_raw"], "ext": update["citation_context_ext"],
                                          "id": update["id"]})
            trans.commit()
        logging.info("Veritabanı başarıyla güncellendi.")
    except Exception as e:
        logging.error(f"Veritabanı güncellenirken bir hata oluştu: {e}")
        trans.rollback()


def main():
    # ... (değişiklik yok)
    setup_logging()
    try:
        engine = create_engine(DB_URL)
    except Exception as e:
        logging.error(f"Veritabanı bağlantısı kurulamadı: {e}")
        return

    prepare_database_columns(engine)
    articles_df, citations_df = fetch_data_to_process(engine)

    if articles_df is not None and citations_df is not None:
        all_updates = []
        logging.info("Tüm makalelerdeki veriler eşleştiriliyor ve güncellemeler toplanıyor...")
        for _, row in tqdm(articles_df.iterrows(), total=articles_df.shape[0], desc="Makaleler Eşleştiriliyor"):
            article_id = row['id']
            tei_content = row['tei_content']
            updates_from_article = process_article_and_collect_updates(tei_content, article_id, citations_df)
            if updates_from_article:
                all_updates.extend(updates_from_article)

        commit_updates_to_db(engine, all_updates)

    logging.info("İşlem tamamlandı.")


if __name__ == "__main__":
    main()