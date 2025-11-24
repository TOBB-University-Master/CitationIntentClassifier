import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Proje modülleri
from config import Config
from dataset import CitationDataset
from generic_model import TransformerClassifier


# ==============================================================================
#                      YARDIMCI FONKSİYONLAR
# ==============================================================================

def log_and_print(message, file_handle):
    """Mesajı hem konsola basar hem de belirtilen dosyaya yazar."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")


def get_flat_predictions(model, data_loader, device):
    """Experiment 1 (Flat) için tahmin fonksiyonu."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def get_hierarchical_predictions(binary_model, multiclass_model, data_loader, device,
                                 binary_encoder, multiclass_encoder, full_label_encoder):
    """Experiment 2 ve 3 (Hiyerarşik) için tahmin fonksiyonu."""
    binary_model.eval()
    multiclass_model.eval()
    all_preds, all_labels = [], []

    # Kritik ID'leri al
    try:
        non_bg_id = binary_encoder.transform(['non-background'])[0]
        bg_orig_id = full_label_encoder.transform(['background'])[0]
    except Exception as e:
        print(f"Encoder hatası: {e}")
        return [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            # 1. Binary Model
            bin_logits = binary_model(input_ids, attention_mask)
            bin_preds = torch.argmax(bin_logits, dim=1)

            final_preds = torch.full_like(bin_preds, fill_value=-1)

            # 2. Uzman Model (Sadece Non-Background Olanlar)
            expert_indices = (bin_preds == non_bg_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                exp_input = input_ids[expert_indices]
                exp_mask = attention_mask[expert_indices]

                multi_logits = multiclass_model(exp_input, exp_mask)
                multi_preds_raw = torch.argmax(multi_logits, dim=1)

                # Uzman çıktısını (0-3) Global ID'ye (0-4) çevir
                cls_names = multiclass_encoder.inverse_transform(multi_preds_raw.cpu().numpy())
                global_ids = full_label_encoder.transform(cls_names)

                final_preds[expert_indices] = torch.tensor(global_ids, device=device)

            # 3. Background Olanlar
            bg_indices = (bin_preds != non_bg_id).nonzero(as_tuple=True)[0]
            final_preds[bg_indices] = bg_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def find_best_trial(model_name, checkpoint_dir):
    """
    Önce Optuna DB'yi kontrol eder, yoksa klasörleri tarar.
    En iyi trial yolunu döndürür.
    """
    model_short = Config.get_model_short_name(model_name)

    # 1. Optuna DB Kontrolü
    # Exp 1 için db ismi: {model}.db
    # Exp 2/3 için db ismi: {model}_hierarchical.db (veya training kodunuza göre değişebilir)

    # Olası DB isimlerini kontrol edelim
    possible_dbs = [
        f"{model_short}.db",
        f"{model_short}_hierarchical.db",
        f"{model_short}_refined.db"
    ]

    found_db = None
    for db_name in possible_dbs:
        db_path = os.path.join(checkpoint_dir, db_name)
        if os.path.exists(db_path):
            found_db = db_path
            break

    if found_db:
        try:
            # Study ismini bulmak zor olabilir, deneme yanılma yapıyoruz
            # Genelde: {model_short}_study veya {model_short}_hiearchical_study
            storage_url = f"sqlite:///{found_db}"
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            if summaries:
                study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
                trial_num = study.best_trial.number
                best_val = study.best_value
                trial_path = os.path.join(checkpoint_dir, model_short, f"trial_{trial_num}")
                if os.path.exists(trial_path):
                    return trial_path, best_val, "Optuna DB"
        except Exception as e:
            print(f"DB Okuma Hatası: {e}")

    # 2. Manuel Tarama (Fallback)
    model_dir = os.path.join(checkpoint_dir, model_short)
    if not os.path.exists(model_dir):
        return None, 0.0, "Not Found"

    # Klasörleri gez, best_model.pt var mı bak (Skor bilemeyiz, sonuncuyu veya ilkini alırız)
    # Gelişmiş: trial_config.json veya log dosyasından skor okuyabiliriz ama şimdilik varlık kontrolü.
    trials = [d for d in os.listdir(model_dir) if d.startswith("trial_")]

    # Basit mantık: trial numarası en büyük olanı (en son yapılanı) veya 0'ı alalım.
    # Burada mantığı geliştirebilirsiniz.
    for trial in sorted(trials, reverse=True):  # Sondan başa
        t_path = os.path.join(model_dir, trial)
        # Exp 1 için
        if os.path.exists(os.path.join(t_path, "best_model.pt")):
            return t_path, 0.0, "Manual Scan (Flat)"
        # Exp 2/3 için
        if os.path.exists(os.path.join(t_path, "binary/best_model.pt")):
            return t_path, 0.0, "Manual Scan (Hierarchical)"

    return None, 0.0, "Failed"


# ==============================================================================
#                      ANA AKIŞ
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument("--experiment_id", type=int, default=None, help="1, 2 or 3")
    parser.add_argument("--prefix_dir", type=str, default=None, help="Opsiyonel: checkpoints klasörünün üst dizini")
    args = parser.parse_args()

    # 1. Config Ayarla
    Config.set_prefix(args.prefix_dir)
    Config.set_experiment(args.experiment_id)
    Config.print_config()

    # Cihaz
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")

    # Sonuç Klasörü
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    log_file_path = os.path.join(Config.RESULTS_DIR, "evaluation_report.txt")

    results = []

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_and_print(f"Değerlendirme Başlıyor - Experiment {Config.EXPERIMENT_ID}", log_file)

        # Test Verisini Yükle
        if not os.path.exists(Config.DATA_PATH_TEST):
            log_and_print(f"HATA: Test verisi bulunamadı: {Config.DATA_PATH_TEST}", log_file)
            return

        test_df = pd.read_csv(Config.DATA_PATH_TEST)
        log_and_print(f"Test Verisi Yüklendi: {len(test_df)} satır.", log_file)

        for model_name in Config.MODELS:
            model_short = Config.get_model_short_name(model_name)
            log_and_print(f"\n{'=' * 40}\nModel Değerlendiriliyor: {model_short}\n{'=' * 40}", log_file)

            # En iyi trial'ı bul
            best_trial_path, best_val_score, source = find_best_trial(model_name, Config.CHECKPOINT_DIR)

            if not best_trial_path:
                log_and_print(f"UYARI: {model_short} için geçerli model bulunamadı. Atlanıyor.", log_file)
                continue

            log_and_print(f"Trial Kaynağı: {source} | Yol: {best_trial_path}", log_file)

            try:
                tokenizer = AutoTokenizer.from_pretrained(best_trial_path)

                # --- EXPERIMENT 1 (FLAT) ---
                if Config.EXPERIMENT_ID == 1:
                    model_path = os.path.join(best_trial_path, "best_model.pt")
                    enc_path = os.path.join(best_trial_path, "label_encoder.pkl")

                    with open(enc_path, "rb") as f:
                        label_encoder = pickle.load(f)
                    label_names = label_encoder.classes_

                    model = TransformerClassifier(model_name, num_labels=len(label_names))
                    model.transformer.resize_token_embeddings(len(tokenizer))
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)

                    # Dataset
                    test_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                                   data_frame=test_df, task=None, include_section_in_input=False)
                    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

                    preds, true_labels = get_flat_predictions(model, test_loader, device)

                # --- EXPERIMENT 2 & 3 (HIERARCHICAL) ---
                else:
                    bin_dir = os.path.join(best_trial_path, "binary")
                    multi_dir = os.path.join(best_trial_path, "multiclass")

                    # 1. Klasör Kontrolü (Eğer binary klasörü yoksa bu bir Flat model olabilir)
                    if not os.path.exists(bin_dir):
                        log_and_print(f"UYARI: '{model_short}' için hiyerarşik klasör yapısı ('binary') bulunamadı.",
                                      log_file)
                        log_and_print(
                            f"       Muhtemelen bu bir FLAT modeldir ama Experiment ID={Config.EXPERIMENT_ID} seçildi.",
                            log_file)
                        continue

                    # 2. Dosya İsmi Bulma (Eski ve Yeni versiyon uyumluluğu için)
                    def find_file(directory, choices):
                        for name in choices:
                            path = os.path.join(directory, name)
                            if os.path.exists(path):
                                return path
                        return None

                    # Binary Encoder Yolu
                    bin_enc_path = find_file(bin_dir, ["label_encoder.pkl", "label_encoder_binary.pkl"])
                    # Binary Model Yolu
                    bin_model_path = find_file(bin_dir, ["best_model.pt", "pytorch_model.bin"])

                    # Multiclass Encoder Yolu
                    multi_enc_path = find_file(multi_dir, ["label_encoder.pkl", "label_encoder_multiclass.pkl"])
                    # Multiclass Model Yolu
                    multi_model_path = find_file(multi_dir, ["best_model.pt", "pytorch_model.bin"])

                    # Eğer dosyalardan biri eksikse hata ver
                    if not all([bin_enc_path, bin_model_path, multi_enc_path, multi_model_path]):
                        log_and_print(f"HATA: Gerekli model dosyaları eksik. Klasör: {best_trial_path}", log_file)
                        continue

                    # 3. Yükleme İşlemleri
                    # Binary Load
                    with open(bin_enc_path, "rb") as f:
                        bin_enc = pickle.load(f)
                    bin_model = TransformerClassifier(model_name, num_labels=len(bin_enc.classes_))
                    bin_model.transformer.resize_token_embeddings(len(tokenizer))
                    bin_model.load_state_dict(torch.load(bin_model_path, map_location=device))
                    bin_model.to(device)

                    # Multiclass Load
                    with open(multi_enc_path, "rb") as f:
                        multi_enc = pickle.load(f)
                    multi_model = TransformerClassifier(model_name, num_labels=len(multi_enc.classes_))
                    multi_model.transformer.resize_token_embeddings(len(tokenizer))
                    multi_model.load_state_dict(torch.load(multi_model_path, map_location=device))
                    multi_model.to(device)

                    # Dataset (Exp 3 için context True)
                    use_context = (Config.EXPERIMENT_ID == 3)
                    test_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                                   data_frame=test_df, task="all", include_section_in_input=use_context)
                    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
                    label_names = test_dataset.label_encoder.classes_

                    preds, true_labels = get_hierarchical_predictions(
                        bin_model, multi_model, test_loader, device,
                        bin_enc, multi_enc, test_dataset.label_encoder
                    )

                # --- Ortak Raporlama ---
                acc = accuracy_score(true_labels, preds)
                f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
                report = classification_report(true_labels, preds, target_names=label_names, zero_division=0)

                log_and_print(f"Test Accuracy: {acc:.4f}", log_file)
                log_and_print(f"Test Macro F1: {f1:.4f}", log_file)
                log_and_print(report, log_file)

                # Confusion Matrix
                cm = confusion_matrix(true_labels, preds)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
                plt.title(f'CM - {model_short} (Exp {Config.EXPERIMENT_ID})')
                plt.ylabel('True');
                plt.xlabel('Pred')
                plt.tight_layout()
                plt.savefig(os.path.join(Config.RESULTS_DIR, f"cm_{model_short}.png"))
                plt.close()

                results.append({"Model": model_short, "Accuracy": acc, "Macro F1": f1})

            except Exception as e:
                log_and_print(f"HATA ({model_short}): {e}", log_file)
                import traceback
                traceback.print_exc()

        # Genel Özet
        if results:
            df_res = pd.DataFrame(results)
            log_and_print("\n\n" + "=" * 40, log_file)
            log_and_print("GENEL SONUÇ TABLOSU", log_file)
            log_and_print(df_res.to_string(index=False), log_file)

            # Karşılaştırma Grafiği
            df_melt = df_res.melt(id_vars="Model", var_name="Metric", value_name="Score")
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="viridis")
            plt.title(f"Model Performans Karşılaştırması (Exp {Config.EXPERIMENT_ID})")
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULTS_DIR, "comparison_chart.png"))
            plt.close()

    print(f"\n✅ İşlem Tamamlandı. Sonuçlar: {Config.RESULTS_DIR}")


if __name__ == "__main__":
    main()