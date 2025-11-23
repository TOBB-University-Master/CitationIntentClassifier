import torch
import os
import json
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torch import Generator
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

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
DATA_PATH = "data/data_v2.csv"
BASE_CHECKPOINT_DIR = "checkpoints_v1"
SEED = 42
BATCH_SIZE = 16

# Sonuçların ve grafiklerin kaydedileceği klasör
RESULTS_DIR = "outputs/experiment_1_1_flat"
# ==============================================================================

def log_and_print(message, file_handle):
    """Mesajı hem konsola basar hem de belirtilen dosyaya yazar."""
    print(message)
    file_handle.write(message + "\n")


def find_best_trial_manually(model_checkpoint_dir):
    """
    Optuna DB olmadan, checkpoint dosyalarındaki 'best_val_acc' değerini okuyarak
    en iyi deneme klasörünü manuel olarak bulur.
    """
    best_trial_path = None
    highest_acc = -1.0
    print(f"'{model_checkpoint_dir}' içindeki denemeler taranıyor...")
    if not os.path.isdir(model_checkpoint_dir):
        print(f"UYARI: Dizin bulunamadı: {model_checkpoint_dir}")
        return None, -1
    for trial_dir_name in os.listdir(model_checkpoint_dir):
        trial_path = os.path.join(model_checkpoint_dir, trial_dir_name)
        if os.path.isdir(trial_path) and trial_dir_name.startswith("trial_"):
            checkpoint_path = os.path.join(trial_path, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                    val_acc = checkpoint.get("best_val_acc", 0.0)
                    if val_acc > highest_acc:
                        highest_acc = val_acc
                        best_trial_path = trial_path
                except Exception as e:
                    print(f"UYARI: {checkpoint_path} dosyası okunurken hata oluştu: {e}")
            else:
                print(f"UYARI: Checkpoint dosyası bulunamadı: {checkpoint_path}")
    return best_trial_path, highest_acc


def get_predictions(model, data_loader, device):
    """
    Modeli değerlendirir ve tahminler ile gerçek etiketleri döndürür.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


def main():
    """
    Ana değerlendirme fonksiyonu.
    """
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
        for model_name in MODELS:
            model_short_name = model_name.split('/')[-1]
            log_and_print("-" * 50, log_file)
            log_and_print(f" değerlendiriliyor: {model_name}", log_file)
            log_and_print("-" * 50, log_file)

            model_checkpoint_dir = os.path.join(BASE_CHECKPOINT_DIR, model_short_name)
            best_trial_dir, best_val_acc = find_best_trial_manually(model_checkpoint_dir)

            if best_trial_dir is None:
                log_and_print(f"HATA: {model_name} için geçerli bir deneme (trial) bulunamadı. Bu model atlanıyor.",
                              log_file)
                continue

            log_and_print(
                f"En iyi deneme (trial) bulundu: {os.path.basename(best_trial_dir)} (Validation Accuracy: {best_val_acc:.4f})",
                log_file)
            log_and_print("Model ve bileşenler yükleniyor...", log_file)
            best_model_path = os.path.join(best_trial_dir, "best_model.pt")
            label_encoder_path = os.path.join(best_trial_dir, "label_encoder.pkl")

            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            label_names = label_encoder.classes_.tolist()
            num_labels = len(label_names)
            tokenizer = AutoTokenizer.from_pretrained(best_trial_dir)
            model = TransformerClassifier(model_name=model_name, num_labels=num_labels)
            model.transformer.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.to(device)

            log_and_print("Test veri seti yeniden oluşturuluyor...", log_file)
            full_dataset = CitationDataset(tokenizer=tokenizer, csv_path=DATA_PATH, max_len=128)
            generator = Generator().manual_seed(SEED)
            train_val_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_val_size
            _, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=generator)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            log_and_print(f"Test için {len(test_dataset)} örnek yüklendi.", log_file)

            log_and_print("Test veri seti üzerinde değerlendirme yapılıyor...", log_file)
            predictions, true_labels = get_predictions(model, test_loader, device)
            accuracy = accuracy_score(true_labels, predictions)
            macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
            report = classification_report(true_labels, predictions, target_names=label_names, zero_division=0)

            log_and_print("\n--- TEST SONUÇLARI ---", log_file)
            log_and_print(f"Accuracy: {accuracy:.4f}", log_file)
            log_and_print(f"Macro F1-Score: {macro_f1:.4f}", log_file)
            log_and_print("Sınıflandırma Raporu:", log_file)
            log_and_print(report, log_file)

            # --- YENİ: CONFUSION MATRIX BÖLÜMÜ ---
            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names)
            plt.title(f'Confusion Matrix - {model_short_name}', fontsize=16)
            plt.ylabel('Gerçek Etiket (True Label)', fontsize=12)
            plt.xlabel('Tahmin Edilen Etiket (Predicted Label)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            cm_plot_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_short_name}.png")
            plt.savefig(cm_plot_path)
            plt.close()  # Bir sonraki döngü için grafiği temizle
            log_and_print(f"\nConfusion matrix kaydedildi: {cm_plot_path}", log_file)
            # --- YENİ BÖLÜM SONU ---

            results.append({
                "model_name": model_short_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            })

        summary_df = pd.DataFrame(results)
        summary_string = summary_df.to_string(index=False)

        log_and_print("\n\n" + "=" * 60, log_file)
        log_and_print("           TÜM MODELLERİN ÖZET SONUÇLARI (Experiment 1.1)", log_file)
        log_and_print("=" * 60, log_file)
        log_and_print(summary_string, log_file)
        log_and_print("=" * 60, log_file)

        print("\nSonuçlar görselleştiriliyor ve kaydediliyor...")
        plt.style.use('seaborn-v0_8-whitegrid')
        df_melted = summary_df.melt(id_vars='model_name', var_name='Metric', value_name='Score')
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric', ax=ax, palette="viridis")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        ax.set_title('Experiment 1.1: Flat Classification Model Karşılaştırması', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model Adı', fontsize=12)
        ax.set_ylabel('Skor', fontsize=12)
        ax.set_ylim(0, 1.0)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        log_and_print(f"\nKarşılaştırma grafiği kaydedildi: {plot_path}", log_file)

    print(f"\nTüm işlemler tamamlandı. Sonuçlar '{RESULTS_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    main()