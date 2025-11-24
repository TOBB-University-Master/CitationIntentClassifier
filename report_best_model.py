import os
import optuna
import pandas as pd
import argparse
import sys


def get_best_models_from_dir(base_dir):
    """
    Verilen ana klasör (örn: _train_004) altındaki tüm checkpoints_v{i}
    klasörlerini tarar ve Optuna veritabanlarındaki en iyi sonuçları raporlar.
    """
    results = []

    # Klasörün varlığını kontrol et
    if not os.path.exists(base_dir):
        print(f"HATA: '{base_dir}' klasörü bulunamadı.")
        return pd.DataFrame()

    print(f"🔍 '{base_dir}' dizini taranıyor...\n")

    # Alt klasörleri gez (checkpoints_v1, v2, v3...)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".db"):
                db_path = os.path.join(root, file)
                storage_url = f"sqlite:///{db_path}"

                # Klasör ismini al (örn: checkpoints_v1)
                folder_name = os.path.basename(root)

                try:
                    # DB içindeki tüm çalışmaları (studies) çek
                    summaries = optuna.study.get_all_study_summaries(storage=storage_url)

                    for summary in summaries:
                        # Eğer deneme yapılmamışsa atla
                        if summary.n_trials == 0:
                            continue

                        best_trial = summary.best_trial

                        # Parametreleri string haline getir (okunabilir olsun)
                        params_str = ", ".join([f"{k}={v}" for k, v in best_trial.params.items()])

                        # Kaydı listeye ekle
                        results.append({
                            "Folder": folder_name,
                            "Model DB": file,
                            "Study Name": summary.study_name,
                            "Best Score": best_trial.value,
                            "Best Trial #": best_trial.number,
                            "N_Trials": summary.n_trials,
                            "Params": params_str,
                            "Start Date": summary.datetime_start.strftime('%Y-%m-%d %H:%M')
                        })

                except Exception as e:
                    print(f"⚠️  Hata ({file}): {e}")

    # Pandas DataFrame oluştur
    if results:
        df = pd.DataFrame(results)
        # Skora göre sırala (Büyükten küçüğe)
        df = df.sort_values(by="Best Score", ascending=False).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Optuna En İyi Modeller Raporlayıcı")
    parser.add_argument("--dir", type=str, default="/Volumes/ULAKBIM/_train_003/checkpoints_v3", help="Taranacak ana eğitim klasörü (örn: _train_004)")
    parser.add_argument("--save", type=str, default="",
                        help="Sonuçları kaydetmek için CSV dosya adı (örn: results.csv)")
    args = parser.parse_args()

    df_results = get_best_models_from_dir(args.dir)

    if not df_results.empty:
        print("=" * 100)
        print(f"🏆 EN İYİ MODELLER LİSTESİ ({args.dir})")
        print("=" * 100)

        # Tabloyu ekrana bas (Params sütunu çok uzunsa kırpabiliriz)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 50)  # Parametreleri kısalt

        print(df_results[['Folder', 'Model DB', 'Best Score', 'Best Trial #', 'N_Trials', 'Start Date']])
        print("\n" + "-" * 100)

        # En iyi 3 modelin parametrelerini detaylı göster
        print("\n🌟 EN İYİ 3 MODEL DETAYI:")
        for i in range(min(3, len(df_results))):
            row = df_results.iloc[i]
            print(f"\n{i + 1}. {row['Model DB']} (Skor: {row['Best Score']:.4f})")
            print(f"   Klasör: {row['Folder']} | Trial: {row['Best Trial #']}")
            print(f"   Parametreler: {row['Params']}")

        # Kaydetme opsiyonu
        if args.save:
            save_path = os.path.join(args.dir, args.save)
            df_results.to_csv(save_path, index=False)
            print(f"\n💾 Rapor kaydedildi: {save_path}")
    else:
        print("❌ Hiçbir sonuç bulunamadı veya veritabanları boş.")


if __name__ == "__main__":
    main()