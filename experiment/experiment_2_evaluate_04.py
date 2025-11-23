import torch
import os
import pickle
import pandas as pd
import optuna
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import Config  # Oluşturduğumuz sınıfı çağırıyoruz
from dataset import CitationDataset
from generic_model import TransformerClassifier


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

            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.full_like(binary_preds, fill_value=-1)

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


def find_best_trial_on_validation(model_name, device):
    """
    Config sınıfındaki yolları kullanarak en iyi trial'ı bulur.
    """
    model_short_name = Config.get_model_short_name(model_name)
    model_checkpoint_dir = Config.get_checkpoint_path(model_name)

    # --- YÖNTEM 1: OPTUNA DB KONTROLÜ ---
    db_path = Config.get_optuna_db_path(model_name)
    storage_url = f"sqlite:///{db_path}"
    study_name = f"{model_short_name}_refined_study"

    if os.path.exists(db_path):
        print(f"   [BİLGİ] Optuna veritabanı bulundu: {db_path}")
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            best_trial = study.best_trial
            best_trial_number = best_trial.number
            best_value = best_trial.value

            print(f"   [DB] En iyi trial (Optuna): #{best_trial_number} (Skor: {best_value:.4f})")

            trial_path = os.path.join(model_checkpoint_dir, f"trial_{best_trial_number}")
            if os.path.exists(trial_path):
                return trial_path, best_value
            else:
                print(f"   [UYARI] DB'deki trial klasörü diskte bulunamadı! Manuel taramaya geçiliyor.")
        except Exception as e:
            print(f"   [HATA] Optuna DB okunamadı ({e}). Manuel taramaya geçiliyor.")

    # --- YÖNTEM 2: MANUEL TARAMA (DB YOKSA) ---
    metric = Config.EVALUATION_METRIC
    print(f"   [BİLGİ] Manuel tarama başlatılıyor... (Metric: {metric})")

    if not os.path.exists(model_checkpoint_dir):
        return None, 0.0

    trial_dirs = [d for d in os.listdir(model_checkpoint_dir) if
                  d.startswith("trial_") and os.path.isdir(os.path.join(model_checkpoint_dir, d))]

    if not trial_dirs:
        return None, 0.0

    # Validation verisini Config'den al
    try:
        val_df = pd.read_csv(Config.DATA_PATH_VAL)
    except FileNotFoundError:
        print(f"HATA: Validation dosyası bulunamadı: {Config.DATA_PATH_VAL}")
        return None, 0.0

    best_trial_path = None
    best_metric_value = -1.0

    for trial_dir_name in tqdm(trial_dirs, desc=f"Scanning trials for {model_short_name}"):
        trial_path = os.path.join(model_checkpoint_dir, trial_dir_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(trial_path)

            # Binary Model
            binary_path = os.path.join(trial_path, "binary")
            with open(os.path.join(binary_path, "label_encoder.pkl"), "rb") as f:
                binary_encoder = pickle.load(f)
            binary_model = TransformerClassifier(model_name=model_name, num_labels=len(binary_encoder.classes_))
            binary_model.transformer.resize_token_embeddings(len(tokenizer))
            binary_model.load_state_dict(torch.load(os.path.join(binary_path, "best_model.pt"), map_location=device))
            binary_model.to(device)

            # Multiclass Model
            multiclass_path = os.path.join(trial_path, "multiclass")
            with open(os.path.join(multiclass_path, "label_encoder.pkl"), "rb") as f:
                multiclass_encoder = pickle.load(f)
            multiclass_model = TransformerClassifier(model_name=model_name, num_labels=len(multiclass_encoder.classes_))
            multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
            multiclass_model.load_state_dict(
                torch.load(os.path.join(multiclass_path, "best_model.pt"), map_location=device))
            multiclass_model.to(device)

            # Validation Loader
            val_dataset = CitationDataset(tokenizer=tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                          data_frame=val_df, task="all")
            full_label_encoder = val_dataset.label_encoder
            val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

            preds, labels = get_hierarchical_predictions(
                binary_model, multiclass_model, val_loader, device,
                binary_encoder, multiclass_encoder, full_label_encoder
            )

            if metric == "accuracy":
                current_score = accuracy_score(labels, preds)
            elif metric == "macro_f1":
                current_score = f1_score(labels, preds, average="macro", zero_division=0)
            else:
                current_score = accuracy_score(labels, preds)

            if current_score > best_metric_value:
                best_metric_value = current_score
                best_trial_path = trial_path

        except Exception as e:
            continue

    return best_trial_path, best_metric_value


def main():
    # Config.ensure_directories() # Class yüklenirken otomatik çalışır ama garanti olsun diye burada da durabilir.

    log_file_path = os.path.join(Config.RESULTS_DIR, "test_summary_report.txt")

    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        log_and_print(f"Kullanılan Cihaz: {device}", log_file)
        log_and_print(f"Değerlendirme Metriği: {Config.EVALUATION_METRIC}", log_file)
        log_and_print(f"Validation Dosyası: {Config.DATA_PATH_VAL}", log_file)
        log_and_print(f"Test Dosyası: {Config.DATA_PATH_TEST}", log_file)

        results = []

        try:
            test_df = pd.read_csv(Config.DATA_PATH_TEST)
            log_and_print(f"Test Verisi Yüklendi: {len(test_df)} satır", log_file)
        except FileNotFoundError:
            log_and_print(f"HATA: Test dosyası bulunamadı: {Config.DATA_PATH_TEST}", log_file)
            return

        for model_name in Config.MODELS:
            model_short_name = Config.get_model_short_name(model_name)
            log_and_print("-" * 50, log_file)
            log_and_print(f" Değerlendiriliyor: {model_name}", log_file)
            log_and_print("-" * 50, log_file)

            # En iyi trial'ı bul
            best_trial_dir, best_val_score = find_best_trial_on_validation(model_name, device)

            if best_trial_dir is None:
                log_and_print(f"HATA: Geçerli bir deneme bulunamadı veya checkpoint yok.", log_file)
                continue

            log_and_print(
                f"Seçilen En İyi Deneme: {os.path.basename(best_trial_dir)} (Ref Score: {best_val_score:.4f})",
                log_file)

            # Modeli Yükle ve Test Et
            log_and_print("Model yükleniyor ve test ediliyor...", log_file)
            try:
                tokenizer = AutoTokenizer.from_pretrained(best_trial_dir)

                binary_path = os.path.join(best_trial_dir, "binary")
                with open(os.path.join(binary_path, "label_encoder.pkl"), "rb") as f:
                    binary_encoder = pickle.load(f)
                binary_model = TransformerClassifier(model_name=model_name, num_labels=len(binary_encoder.classes_))
                binary_model.transformer.resize_token_embeddings(len(tokenizer))
                binary_model.load_state_dict(
                    torch.load(os.path.join(binary_path, "best_model.pt"), map_location=device))
                binary_model.to(device)

                multiclass_path = os.path.join(best_trial_dir, "multiclass")
                with open(os.path.join(multiclass_path, "label_encoder.pkl"), "rb") as f:
                    multiclass_encoder = pickle.load(f)
                multiclass_model = TransformerClassifier(model_name=model_name,
                                                         num_labels=len(multiclass_encoder.classes_))
                multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
                multiclass_model.load_state_dict(
                    torch.load(os.path.join(multiclass_path, "best_model.pt"), map_location=device))
                multiclass_model.to(device)

                test_dataset = CitationDataset(tokenizer=tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                               data_frame=test_df, task="all")
                full_label_encoder = test_dataset.label_encoder
                label_names = full_label_encoder.classes_.tolist()

                test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

                predictions, true_labels = get_hierarchical_predictions(
                    binary_model, multiclass_model, test_loader, device,
                    binary_encoder, multiclass_encoder, full_label_encoder
                )

                accuracy = accuracy_score(true_labels, predictions)
                macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
                report = classification_report(true_labels, predictions, target_names=label_names, zero_division=0)

                log_and_print("\n--- TEST SONUÇLARI (HIYERARŞİK) ---", log_file)
                log_and_print(f"Accuracy: {accuracy:.4f}", log_file)
                log_and_print(f"Macro F1-Score: {macro_f1:.4f}", log_file)
                log_and_print("Sınıflandırma Raporu:", log_file)
                log_and_print(report, log_file)

                # Confusion Matrix
                cm = confusion_matrix(true_labels, predictions)
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
                plt.title(f'Confusion Matrix (Test) - {model_short_name}', fontsize=16)
                plt.ylabel('Gerçek Etiket')
                plt.xlabel('Tahmin Edilen Etiket')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(Config.RESULTS_DIR, f"cm_{model_short_name}.png"))
                plt.close()

                results.append({
                    "model_name": model_short_name,
                    "test_accuracy": accuracy,
                    "test_macro_f1": macro_f1
                })

            except Exception as e:
                log_and_print(f"HATA: Test sırasında beklenmedik hata oluştu ({model_short_name}): {e}", log_file)
                continue

        if results:
            summary_df = pd.DataFrame(results)
            log_and_print("\n\n" + "=" * 60, log_file)
            log_and_print("     TÜM MODELLERİN TEST ÖZETİ", log_file)
            log_and_print("=" * 60, log_file)
            log_and_print(summary_df.to_string(index=False), log_file)

            plt.style.use('seaborn-v0_8-whitegrid')
            df_melted = summary_df.melt(id_vars='model_name', var_name='Metric', value_name='Score')
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric', ax=ax, palette="viridis")
            for p in ax.patches:
                if p.get_height() > 0:
                    ax.annotate(format(p.get_height(), '.4f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', xytext=(0, 9),
                                textcoords='offset points', fontsize=10)
            ax.set_title('Model Karşılaştırması (Test Seti)', fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1.05)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULTS_DIR, "test_model_comparison.png"))
            plt.close()

    print(f"\nTüm işlemler tamamlandı. Sonuçlar '{Config.RESULTS_DIR}' klasöründe.")


if __name__ == "__main__":
    main()