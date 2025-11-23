import torch
import os
import json
import pickle
import pandas as pd
import optuna  # Optuna eklendi
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Yerel projenizdeki dosyalara referans (import)
from dataset import CitationDataset
from generic_model import TransformerClassifier

# ==============================================================================
#                      *** YAPILANDIRMA ***
# ==============================================================================
MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]

DATA_PATH_VAL = "data/data_v2_val_ext.csv"
DATA_PATH_TEST = "data/data_v2_test_ext.csv"

BASE_CHECKPOINT_DIR = "checkpoints_v3"
BATCH_SIZE = 16
RESULTS_DIR = "outputs_test/experiment_3_context_aware_hierarchical"


# ==============================================================================

def log_and_print(message, file_handle):
    """Mesajı hem konsola basar hem de belirtilen dosyaya yazar."""
    print(message)
    file_handle.write(message + "\n")


def get_hierarchical_predictions(binary_model, multiclass_model, data_loader, device, binary_encoder,
                                 multiclass_encoder, full_label_encoder):
    """
    İkili ve uzman modelleri kullanarak hiyerarşik tahmin yapar.
    """
    binary_model.eval()
    multiclass_model.eval()
    all_preds, all_labels = [], []

    non_background_binary_id = binary_encoder.transform(['non-background'])[0]
    background_orig_id = full_label_encoder.transform(['background'])[0]

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            # 1. Aşama: Binary Model
            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.full_like(binary_preds, fill_value=-1)

            # 2. Aşama: Uzman Model
            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                expert_input_ids = input_ids[expert_indices]
                expert_attention_mask = attention_mask[expert_indices]
                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)

                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = full_label_encoder.transform(multiclass_class_names)

                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def get_best_trial_from_optuna(model_name, checkpoint_dir):
    """
    Optuna veritabanını (.db) kontrol eder ve en iyi denemeyi (trial) döndürür.
    """
    model_short_name = model_name.split('/')[-1]

    # Eğitim kodundaki isimlendirme standardı: {model}_hierarchical.db
    db_path = os.path.join(checkpoint_dir, f"{model_short_name}_hierarchical.db")

    if not os.path.exists(db_path):
        return None, None, "DB_NOT_FOUND"

    storage_url = f"sqlite:///{db_path}"
    # Eğitim kodundaki study ismi: {model}_hiearchical_study (Training kodundaki yazıma sadık kalındı)
    study_name = f"{model_short_name}_hiearchical_study"

    try:
        print(f"Optuna DB bulundu, okunuyor: {db_path}")
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        best_trial_num = study.best_trial.number
        best_value = study.best_value

        trial_path = os.path.join(checkpoint_dir, model_short_name, f"trial_{best_trial_num}")

        if not os.path.exists(trial_path):
            print(f"UYARI: DB'de en iyi trial {best_trial_num} görünüyor ama klasör bulunamadı: {trial_path}")
            return None, None, "TRIAL_DIR_MISSING"

        return trial_path, best_value, "SUCCESS"

    except Exception as e:
        print(f"Optuna DB okuma hatası ({model_short_name}): {e}")
        return None, None, "DB_READ_ERROR"


def find_best_trial_by_scanning(model_name, model_checkpoint_dir, device):
    """
    Eğer DB yoksa, klasörleri tarayarak ve validasyon yaparak en iyiyi bulur.
    (Eski Yöntem)
    """
    best_trial_path = None
    highest_val_acc = -1.0

    model_short_name = model_name.split('/')[-1]
    model_specific_dir = os.path.join(model_checkpoint_dir, model_short_name)

    if not os.path.isdir(model_specific_dir):
        return None, -1.0

    trial_dirs = [d for d in os.listdir(model_specific_dir) if
                  d.startswith("trial_") and os.path.isdir(os.path.join(model_specific_dir, d))]

    print(f"'{model_specific_dir}' içinde {len(trial_dirs)} deneme manuel taranıyor...")

    # Validation verisini RAM'e yükle
    if not os.path.exists(DATA_PATH_VAL):
        print(f"HATA: Validation dosyası bulunamadı: {DATA_PATH_VAL}")
        return None, -1.0

    val_data_in_ram = pd.read_csv(DATA_PATH_VAL)

    for trial_dir_name in tqdm(trial_dirs, desc=f"Scanning trials for {model_short_name}"):
        trial_path = os.path.join(model_specific_dir, trial_dir_name)
        try:
            # Dosya yolları kontrolü
            binary_model_path = os.path.join(trial_path, "binary/best_model.pt")
            binary_encoder_path = os.path.join(trial_path, "binary/label_encoder_binary.pkl")
            multiclass_model_path = os.path.join(trial_path, "multiclass/best_model.pt")
            multiclass_encoder_path = os.path.join(trial_path, "multiclass/label_encoder_multiclass.pkl")

            if not all(os.path.exists(p) for p in
                       [binary_model_path, binary_encoder_path, multiclass_model_path, multiclass_encoder_path]):
                continue

            tokenizer = AutoTokenizer.from_pretrained(trial_path)

            # İkili model yükle
            with open(binary_encoder_path, "rb") as f:
                binary_encoder = pickle.load(f)
            binary_model = TransformerClassifier(model_name=model_name, num_labels=len(binary_encoder.classes_))
            binary_model.transformer.resize_token_embeddings(len(tokenizer))
            binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
            binary_model.to(device)

            # Uzman model yükle
            with open(multiclass_encoder_path, "rb") as f:
                multiclass_encoder = pickle.load(f)
            multiclass_model = TransformerClassifier(model_name=model_name, num_labels=len(multiclass_encoder.classes_))
            multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
            multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
            multiclass_model.to(device)

            # Dataset oluştur
            val_dataset = CitationDataset(
                tokenizer=tokenizer,
                mode="labeled",
                data_frame=val_data_in_ram,
                task='all',
                include_section_in_input=True
            )
            full_label_encoder = val_dataset.label_encoder

            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

            preds, labels = get_hierarchical_predictions(
                binary_model, multiclass_model, val_loader, device,
                binary_encoder, multiclass_encoder, full_label_encoder
            )
            current_val_acc = accuracy_score(labels, preds)

            if current_val_acc > highest_val_acc:
                highest_val_acc = current_val_acc
                best_trial_path = trial_path

        except Exception as e:
            print(f"HATA: {trial_dir_name} işlenirken hata oluştu: {e}")
            continue

    return best_trial_path, highest_val_acc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = os.path.join(RESULTS_DIR, "summary_report.txt")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        log_and_print(f"Kullanılan Cihaz: {device}\n", log_file)

        results = []

        # Test verisini RAM'e yükle
        if not os.path.exists(DATA_PATH_TEST):
            log_and_print(f"KRİTİK HATA: Test dosyası bulunamadı: {DATA_PATH_TEST}", log_file)
            return
        test_data_in_ram = pd.read_csv(DATA_PATH_TEST)
        log_and_print(f"Test veri seti yüklendi: {len(test_data_in_ram)} satır.", log_file)

        for model_name in MODELS:
            model_short_name = model_name.split('/')[-1]
            log_and_print("-" * 50, log_file)
            log_and_print(f"DEĞERLENDİRİLİYOR: {model_name}", log_file)
            log_and_print("-" * 50, log_file)

            best_trial_dir = None
            source_method = "Unknown"
            best_score = 0.0

            # 1. YÖNTEM: Optuna DB Kontrolü
            log_and_print("Yöntem 1: Optuna veritabanı kontrol ediliyor...", log_file)
            trial_dir_optuna, score_optuna, status = get_best_trial_from_optuna(model_name, BASE_CHECKPOINT_DIR)

            if status == "SUCCESS":
                best_trial_dir = trial_dir_optuna
                best_score = score_optuna
                source_method = "Optuna DB"
                log_and_print(
                    f"Optuna DB üzerinden en iyi trial bulundu: {os.path.basename(best_trial_dir)} (Score: {best_score:.4f})",
                    log_file)
            else:
                log_and_print(f"Optuna DB kullanılamadı ({status}). Yöntem 2'ye geçiliyor...", log_file)

                # 2. YÖNTEM: Manuel Tarama (Fallback)
                log_and_print("Yöntem 2: Klasör taraması ve Validation hesaplaması başlatılıyor...", log_file)
                trial_dir_scan, score_scan = find_best_trial_by_scanning(model_name, BASE_CHECKPOINT_DIR, device)

                if trial_dir_scan:
                    best_trial_dir = trial_dir_scan
                    best_score = score_scan
                    source_method = "Manual Scan"
                    log_and_print(
                        f"Tarama sonucu en iyi trial bulundu: {os.path.basename(best_trial_dir)} (Val Acc: {best_score:.4f})",
                        log_file)

            if best_trial_dir is None:
                log_and_print(f"HATA: {model_name} için geçerli bir deneme bulunamadı. Atlanıyor.", log_file)
                continue

            log_and_print("Model ve bileşenler yükleniyor...", log_file)

            try:
                tokenizer = AutoTokenizer.from_pretrained(best_trial_dir)

                # İkili model bileşenleri
                binary_model_path = os.path.join(best_trial_dir, "binary/best_model.pt")
                binary_encoder_path = os.path.join(best_trial_dir, "binary/label_encoder_binary.pkl")
                with open(binary_encoder_path, "rb") as f:
                    binary_encoder = pickle.load(f)
                binary_model = TransformerClassifier(model_name=model_name, num_labels=len(binary_encoder.classes_))
                binary_model.transformer.resize_token_embeddings(len(tokenizer))
                binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
                binary_model.to(device)

                # Uzman model bileşenleri
                multiclass_model_path = os.path.join(best_trial_dir, "multiclass/best_model.pt")
                multiclass_encoder_path = os.path.join(best_trial_dir, "multiclass/label_encoder_multiclass.pkl")
                with open(multiclass_encoder_path, "rb") as f:
                    multiclass_encoder = pickle.load(f)
                multiclass_model = TransformerClassifier(model_name=model_name,
                                                         num_labels=len(multiclass_encoder.classes_))
                multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
                multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
                multiclass_model.to(device)

                # TEST Dataset Hazırlığı
                test_dataset = CitationDataset(
                    tokenizer=tokenizer,
                    mode="labeled",
                    data_frame=test_data_in_ram,
                    task='all',
                    include_section_in_input=True
                )
                full_label_encoder = test_dataset.label_encoder
                label_names = full_label_encoder.classes_.tolist()

                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

                log_and_print("Hiyerarşik (Context-Aware) TEST süreci başlatılıyor...", log_file)
                predictions, true_labels = get_hierarchical_predictions(
                    binary_model, multiclass_model, test_loader, device,
                    binary_encoder, multiclass_encoder, full_label_encoder
                )

                accuracy = accuracy_score(true_labels, predictions)
                macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
                report = classification_report(true_labels, predictions, target_names=label_names, zero_division=0)

                log_and_print(f"\n--- TEST SONUÇLARI ({model_short_name}) ---", log_file)
                log_and_print(f"Seçilen Trial Kaynağı: {source_method}", log_file)
                log_and_print(f"Test Accuracy: {accuracy:.4f}", log_file)
                log_and_print(f"Test Macro F1: {macro_f1:.4f}", log_file)
                log_and_print("Sınıflandırma Raporu:", log_file)
                log_and_print(report, log_file)

                # Confusion Matrix
                cm = confusion_matrix(true_labels, predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
                plt.title(f'Confusion Matrix - {model_short_name}', fontsize=16)
                plt.ylabel('Gerçek', fontsize=12)
                plt.xlabel('Tahmin', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                cm_plot_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_short_name}.png")
                plt.savefig(cm_plot_path)
                plt.close()

                results.append({"model_name": model_short_name, "accuracy": accuracy, "macro_f1": macro_f1})

            except Exception as e:
                log_and_print(f"HATA: {model_name} değerlendirilirken hata oluştu: {e}", log_file)
                continue

        # Özet Tablo ve Grafik
        if results:
            summary_df = pd.DataFrame(results)
            summary_string = summary_df.to_string(index=False)
            log_and_print("\n\n" + "=" * 60, log_file)
            log_and_print("     TÜM MODELLERİN ÖZET SONUÇLARI", log_file)
            log_and_print("=" * 60, log_file)
            log_and_print(summary_string, log_file)

            print("\nGrafikler oluşturuluyor...")
            plt.style.use('seaborn-v0_8-whitegrid')
            df_melted = summary_df.melt(id_vars='model_name', var_name='Metric', value_name='Score')
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric', ax=ax, palette="viridis")
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center',
                            va='center', xytext=(0, 9), textcoords='offset points')
            ax.set_title('Context-Aware Hierarchical Model Karşılaştırması', fontsize=16, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('Skor', fontsize=12)
            ax.set_ylim(0, 1.0)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
            plt.savefig(plot_path)
            plt.close()
            log_and_print(f"\nGrafik kaydedildi: {plot_path}", log_file)

    print(f"\nİşlem tamamlandı. Sonuçlar: {RESULTS_DIR}")


if __name__ == "__main__":
    main()