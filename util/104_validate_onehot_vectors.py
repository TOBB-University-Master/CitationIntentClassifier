import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

"""
    Bu script data_v2_test_one_hot_104.csv dosyasındaki değerlere bakarak acc ve makro f1 değerlerini karşılaştırıyor.
    
    Bu değerler teze konulan düz ve hiyerarşik yapıdaki normal bağlam ile eğitilmiş model sonuçları ile aynı olmalıdır
"""

# Sabit Sınıf Sırası (generate_vectors.py ile aynı olmalı)
TARGET_ORDER = ['background', 'basis', 'support', 'differ', 'discuss']


def validate_file(file_path):
    print(f"\n{'=' * 60}")
    print(f"DOSYA İNCELENİYOR: {os.path.basename(file_path)}")
    print(f"{'=' * 60}")

    if not os.path.exists(file_path):
        print(f"HATA: Dosya bulunamadı -> {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Kayıt Sayısı: {len(df)}")

    # 1. Gerçek Etiketleri Hazırla
    # Küçük harfe çevir ve index'e dönüştür
    label_to_id = {label: i for i, label in enumerate(TARGET_ORDER)}

    # Etiket temizliği (boşluk alma, küçük harf)
    if 'true_label' not in df.columns:
        print("HATA: 'true_label' sütunu bulunamadı.")
        return

    # Pandas map işlemi ile etiketleri sayıya çeviriyoruz
    y_true = df['true_label'].astype(str).str.lower().str.strip().map(label_to_id)

    # Eşleşmeyen etiket var mı kontrol et
    if y_true.isna().any():
        print("UYARI: Tanımsız etiketler bulundu (TARGET_ORDER harici). Bu satırlar atlanıyor.")
        invalid_count = y_true.isna().sum()
        print(f"Atlanan satır sayısı: {invalid_count}")
        # Filtreleme
        valid_indices = ~y_true.isna()
        y_true = y_true[valid_indices].astype(int).values
        df = df[valid_indices].reset_index(drop=True)
    else:
        y_true = y_true.astype(int).values

    # 2. CSV İçindeki Modelleri Otomatik Tespit Et
    # Sütun adı formatı: {model_prefix}_{label} (örn: bert-base_1_background)
    # Strateji: Sonu "_background" ile biten sütunları bulup prefix'i alacağız.
    model_prefixes = set()
    for col in df.columns:
        if col.endswith("_background"):
            prefix = col[:-11]  # "_background" (11 karakter) çıkar
            model_prefixes.add(prefix)

    if not model_prefixes:
        print("UYARI: Model sütunları bulunamadı (Format: modelname_expid_background ...)")
        return

    print(f"Tespit Edilen Modeller ({len(model_prefixes)} adet): {', '.join(sorted(model_prefixes))}\n")

    # Sonuçları saklamak için liste
    results = []

    # 3. Her Model İçin Hesaplama Yap
    for prefix in sorted(model_prefixes):
        # Bu modelin 5 sınıfına ait sütun isimlerini oluştur
        cols = [f"{prefix}_{label}" for label in TARGET_ORDER]

        # Sütunların hepsi var mı kontrol et
        if not all(col in df.columns for col in cols):
            print(f"UYARI: {prefix} için eksik sütunlar var. Atlanıyor.")
            continue

        # One-Hot Matrisini Al (N, 5)
        # argmax ile en yüksek değeri 1 olan indexi bul (yani tahmin edilen sınıf ID'si)
        y_pred_probs = df[cols].values
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Kontrol: Eğer bir satırdaki tüm değerler 0 ise (tahmin yoksa)?
        # argmax hepsi 0 olduğunda 0 (ilk sınıf) döndürür. Bunu kontrol etmek isteyebiliriz.
        # Bu senaryoda modelin output vermediği durumları analiz etmek için:
        sums = y_pred_probs.sum(axis=1)
        zeros_count = (sums == 0).sum()

        # Metrikler
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        results.append({
            "Model": prefix,
            "Accuracy": acc,
            "Macro F1": f1,
            "Boş Tahmin (00000)": zeros_count
        })

    # 4. Tabloyu Yazdır
    results_df = pd.DataFrame(results)
    # Okunabilirlik için sırala (Macro F1'e göre azalan)
    if not results_df.empty:
        results_df = results_df.sort_values(by="Macro F1", ascending=False)
        print(results_df.to_string(index=False, float_format="%.4f"))
    else:
        print("Hesaplanacak model bulunamadı.")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate One-Hot CSV Files")
    parser.add_argument("--file_suffix", type=str, default="104", help="File suffix used in generation (e.g. 104)")
    args = parser.parse_args()

    # İşlenecek dosyalar
    splits = ["train", "val", "test"]

    for split in splits:
        filename = f"data/data_v2_{split}_one_hot_{args.file_suffix}.csv"
        # Eğer current directory'de yoksa, belki data/ klasöründedir diye kontrol edilebilir
        # ama generate scripti current directory'e kaydettiği için oraya bakıyoruz.
        validate_file(filename)


if __name__ == "__main__":
    main()