import os
import optuna
import sys

# ==============================================================================
#           *** train_v3.py'DEN ALINAN AYARLAR ***
# ==============================================================================
MODEL_NAMES = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]  #
DATA_OUTPUT_PATH = "checkpoints_v3_trials/"
# ==============================================================================

def report_best_per_model():
    """
    Tüm modeller için Optuna .db dosyalarını tarar, her modelin en iyi denemesini
    ve o denemenin klasör yolunu sıralı bir liste halinde yazdırır.
    """
    # train_v3.py'deki mantığa dayanarak
    print(f"Veritabanları '{DATA_OUTPUT_PATH}' klasöründe aranıyor...")

    all_results = []

    for model_name in MODEL_NAMES:  #
        model_short_name = model_name.split('/')[-1]
        study_name = f"{model_short_name}_hiearchical_study"  #

        # train_v3.py'deki veritabanı yolu
        db_filename = f"{model_short_name}_hierarchical.db"
        db_path = os.path.join(DATA_OUTPUT_PATH, db_filename)  #
        storage_path = f"sqlite:///{db_path}"  #

        if not os.path.exists(db_path):
            print(f"\nUyarı: '{model_name}' için veritabanı bulunamadı. Atlaniyor.")
            print(f"  Aranan yol: {db_path}")
            continue

        try:
            # Optuna çalışmasını yükle
            study = optuna.load_study(study_name=study_name, storage=storage_path)  #

            if not study.trials:
                print(f"\nUyarı: '{model_name}' çalışması bulundu ancak hiç deneme (trial) yok. Atlaniyor.")
                continue

            best_trial = study.best_trial
            model_best_score = best_trial.value
            model_best_trial_num = best_trial.number

            result = {
                "model_name": model_name,
                "best_score_f1": model_best_score,
                "best_trial_num": model_best_trial_num,
                "params": best_trial.params
            }
            all_results.append(result)

        except Exception as e:
            print(f"\nHATA: '{model_name}' çalışması yüklenirken bir hata oluştu: {e}")

    # --- Sonuçları Raporla ---

    if not all_results:
        print(f"HATA: '{DATA_OUTPUT_PATH}' içinde hiçbir model sonucu bulunamadı.")
        return

    print("\n" + "=" * 70)
    print("      TÜM MODELLERİN EN İYİ DENEME (TRIAL) SONUÇLARI (F1'e göre)")
    print("=" * 70)

    # Sonuçları F1 skoruna göre sırala
    all_results.sort(key=lambda x: x['best_score_f1'], reverse=True)

    for i, res in enumerate(all_results):
        print(f"\n{i + 1}. Model: {res['model_name']}")
        print(f"   En İyi F1 Skoru: {res['best_score_f1']:.5f}")
        print(f"   Deneme (Trial) No: {res['best_trial_num']}")
        print(f"   Parametreler: {res['params']}")

        # --- Her model için özel klasör yolunu burada oluştur ---
        model_short_name = res['model_name'].split('/')[-1]
        # train_v3.py'deki deneme klasörü yolu
        winner_path = os.path.join(DATA_OUTPUT_PATH, f"trial_{res['best_trial_num']}_{model_short_name}/")  #

        print("\n   İndirilecek Klasör Yolu:")
        print(f"   --------------------------------------------------")
        print(f"   {winner_path}")
        print(f"   --------------------------------------------------")

    print("\n" + "=" * 70)
    print("Raporlama tamamlandı.")


if __name__ == "__main__":
    report_best_per_model()