import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm  # İsteğe bağlı, veritabanından çekme işlemini izlemek için


"""
    NOT: BENI OKU!!!
    
    Bu dosya modellerin citation_intent tahmin doğruluklarını (accuracy)
    CSV dosyasındaki gerçek değerlerle karşılaştırarak hesaplar ve raporlar.
    
    Ancak test verisindeki bazı atıflar (citation_id) veritabanında
    bulunmayabilir. Bu durumu düzeltmek için API üzerinden eksik atıflar
    veritabanına eklenmelidir !!!
    
"""

# Sabitler
DB_URL = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
CSV_FILE_PATH = "data/data_v2_test.csv"

USER_MAPPING = {
    "gemini-2.5-flash-k0": "48ed0fcf-4e78-4913-b96b-d942646d34h1",
    "gemini-2.5-flash-k1": "48ed0fcf-4e78-4913-b96b-d942646d34a1",
    "gemini-2.5-flash-k2": "48ed0fcf-4e78-4913-b96b-d942646d34a2",
    "gemini-2.5-flash-k5": "48ed0fcf-4e78-4913-b96b-d942646d34a5",
    "gemini-2.5-pro-k0": "48ed0fcf-4e78-4913-b96b-d942646d3420",
    "chatgpt-4o-k0": "48ed0fcf-4e78-4913-b96b-d942646d34c1",
    "chatgpt-4o-k1": "48ed0fcf-4e78-4913-b96b-d942646d34a6",
    "chatgpt-4o-k2": "48ed0fcf-4e78-4913-b96b-d942646d34a7",
    "chatgpt-4o-k5": "48ed0fcf-4e78-4913-b96b-d942646d3410",
    "dspy-gemini": "48ed0fcf-4e78-4913-b96b-d942646d34d1",
}


def calculate_citation_intent_accuracy():
    """
    CSV dosyasındaki gerçek değerlerle veritabanındaki kullanıcı tahminlerini
    karşılaştırarak accuracy (doğruluk) değerlerini hesaplar.
    """
    try:
        # 1. Ground Truth (Gerçek Değerler) Verisini Yükleme
        df_ground_truth = pd.read_csv(CSV_FILE_PATH)
        df_ground_truth = df_ground_truth[['id', 'citation_intent']].rename(
            columns={'id': 'citation_id_gt', 'citation_intent': 'intent_gt'}
        )
        # Birleştirme (merge) işleminde tür tutarlılığını sağlamak için ID'leri string'e çeviriyoruz.
        df_ground_truth['citation_id_gt'] = df_ground_truth['citation_id_gt'].astype(str)
        print(f"Gerçek değerler yüklendi. Toplam {len(df_ground_truth)} kayıt.")

        # 2. Veritabanı Bağlantısını Kurma
        engine = create_engine(DB_URL)
        print("Veritabanı bağlantısı kuruldu.")

        # 3. Her Bir Kullanıcı/Model İçin Tahminleri Çekme ve Accuracy Hesaplama
        accuracy_results = {}

        # tqdm ile döngüyü sararak ilerlemeyi gösteriyoruz.
        for model_name, user_id in tqdm(USER_MAPPING.items(), desc="Accuracy hesaplanıyor"):
            # Veritabanı sorgusu
            query = f"""
            SELECT citation_id, citation_intent
            FROM cec_citation_intent
            WHERE user_id = '{user_id}'
            """

            # Veritabanından tahminleri çekme
            try:
                df_predictions = pd.read_sql(query, engine)
            except Exception as e:
                print(f"\nHata: '{model_name}' için veritabanı sorgusu başarısız oldu. Hata: {e}")
                continue

            if df_predictions.empty:
                print(f"\nUyarı: '{model_name}' ({user_id}) için tahmin bulunamadı.")
                accuracy_results[model_name] = {'Accuracy': 0.0, 'Total Predictions': 0, 'Matched Citations': 0}
                continue

            # Sütunları yeniden adlandırma ve tür dönüştürme
            df_predictions.rename(
                columns={'citation_intent': 'intent_pred'},
                inplace=True
            )
            df_predictions['citation_id'] = df_predictions['citation_id'].astype(str)

            # Ground Truth ve Tahminleri Birleştirme (Merge)
            # Yalnızca hem ground truth hem de tahmini olan kayıtları alıyoruz (inner join)
            df_merged = pd.merge(
                df_ground_truth,
                df_predictions,
                left_on='citation_id_gt',
                right_on='citation_id',
                how='inner'
            )

            if df_merged.empty:
                accuracy_results[model_name] = {'Accuracy': 0.0, 'Total Predictions': len(df_predictions),
                                                'Matched Citations': 0}
                continue

            # Accuracy Hesaplama: GT ve Tahmin aynı mı?
            correct_predictions = (df_merged['intent_gt'] == df_merged['intent_pred']).sum()
            total_matches = len(df_merged)
            accuracy = correct_predictions / total_matches if total_matches > 0 else 0.0

            accuracy_results[model_name] = {
                'Accuracy': accuracy,
                'Total Matches': total_matches,
                'Correct Predictions': correct_predictions
            }

        # 4. Sonuçları Görüntüleme
        df_results = pd.DataFrame.from_dict(accuracy_results, orient='index')
        df_results.index.name = 'Model Name'

        print("\n" + "=" * 50)
        print("Accuracy Sonuçları")
        print("=" * 50)
        print(df_results.sort_values(by='Accuracy', ascending=False).to_markdown())
        print("=" * 50)
        print(f"Not: 'Total Matches' sütunu, hem CSV'de hem de veritabanında karşılığı bulunan atıf sayısını gösterir.")

    except FileNotFoundError:
        print(f"Hata: '{CSV_FILE_PATH}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {e}")


if __name__ == "__main__":
    calculate_citation_intent_accuracy()