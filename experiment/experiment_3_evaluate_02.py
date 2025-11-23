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
from tqdm import tqdm

# Yerel projenizdeki dosyalara referans (import)
from dataset import CitationDataset
from generic_model import TransformerClassifier

# ==============================================================================
#                      *** YAPILANDIRMA (train_v3.py ile UYUMLU) ***
# ==============================================================================
MODELS = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]
# train_v3.py'den
DATA_PATH = "data/data_v3.csv"
# train_v3.py'deki DATA_OUTPUT_PATH
BASE_CHECKPOINT_DIR = "checkpoints_v3"
SEED = 42
BATCH_SIZE = 16
# Sonuç klasörü v3 olarak güncellendi
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
    (Bu fonksiyon train_v3.py'nin mantığıyla zaten uyumludur)
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


def find_best_trial_by_rerunning_validation(model_name, model_checkpoint_dir, device):
    """
    Her bir denemeyi (trial) doğrulama seti üzerinde yeniden çalıştırarak en iyi
    birleşik doğruluğa sahip denemeyi bulur.

    *** train_v3.py'nin FARKLI KLASÖR YAPISINA GÖRE GÜNCELLENDİ ***
    (Yapı: {BASE_CHECKPOINT_DIR}/{model_short_name}/trial_{num}/)
    """
    best_trial_path = None
    highest_val_acc = -1.0

    model_short_name = model_name.split('/')[-1]
    # train_v3.py yapısına uygun modelin ana klasör yolu
    model_specific_dir = os.path.join(model_checkpoint_dir, model_short_name)

    if not os.path.isdir(model_specific_dir):
        print(f"Uyarı: '{model_specific_dir}' bulunamadı. {model_short_name} için denemeler atlanıyor.")
        return None, -1.0

    # Modelin ana klasörü içindeki 'trial_...' ile başlayan deneme klasörlerini bul
    trial_dirs = [d for d in os.listdir(model_specific_dir) if
                  d.startswith("trial_") and os.path.isdir(os.path.join(model_specific_dir, d))]

    print(f"'{model_specific_dir}' içinde {len(trial_dirs)} deneme taranıyor...")

    for trial_dir_name in tqdm(trial_dirs, desc=f"Finding best trial for {model_short_name}"):
        # Denemenin tam yolu (örn: checkpoints_v3/bert-base-turkish-cased/trial_1)
        trial_path = os.path.join(model_specific_dir, trial_dir_name)
        try:
            # Gerekli dosyaların varlığını kontrol et
            binary_model_path = os.path.join(trial_path, "binary/best_model.pt")
            binary_encoder_path = os.path.join(trial_path, "binary/label_encoder_binary.pkl")
            multiclass_model_path = os.path.join(trial_path, "multiclass/best_model.pt")
            multiclass_encoder_path = os.path.join(trial_path, "multiclass/label_encoder_multiclass.pkl")

            if not all(os.path.exists(p) for p in
                       [binary_model_path, binary_encoder_path, multiclass_model_path, multiclass_encoder_path,
                        trial_path]):
                # print(f"Atlanıyor: {trial_dir_name} eksik dosyalara sahip.")
                continue

            tokenizer = AutoTokenizer.from_pretrained(trial_path)

            # İkili modeli yükle
            with open(binary_encoder_path, "rb") as f:
                binary_encoder = pickle.load(f)
            binary_model = TransformerClassifier(model_name=model_name, num_labels=len(binary_encoder.classes_))
            binary_model.transformer.resize_token_embeddings(len(tokenizer))
            binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
            binary_model.to(device)

            # Çok sınıflı (uzman) modeli yükle
            with open(multiclass_encoder_path, "rb") as f:
                multiclass_encoder = pickle.load(f)
            multiclass_model = TransformerClassifier(model_name=model_name, num_labels=len(multiclass_encoder.classes_))
            multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
            multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
            multiclass_model.to(device)

            # Doğrulama (Validation) setini train_v3.py ile aynı ayarlarla yeniden oluştur
            # (Context-Aware olarak)
            full_dataset = CitationDataset(tokenizer=tokenizer, csv_path=DATA_PATH, max_len=256, task=None,
                                           include_section_in_input=True)
            full_label_encoder = full_dataset.label_encoder
            generator = Generator().manual_seed(SEED)

            # 80% Train/Val
            train_val_size = int(0.8 * len(full_dataset))
            train_val_dataset, _ = random_split(full_dataset, [train_val_size, len(full_dataset) - train_val_size],
                                                generator=generator)

            # Kalan 80%'in 85%'i Train, 15%'i Val
            # train_v3.py ve evaluate script'lerindeki mantıkla aynı
            train_size = int(0.85 * len(train_val_dataset))
            val_size = len(train_val_dataset) - train_size
            # Sadece validation setini (ikinci parçayı) al
            _, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            preds, labels = get_hierarchical_predictions(
                binary_model, multiclass_model, val_loader, device,
                binary_encoder, multiclass_encoder, full_label_encoder
            )
            current_val_acc = accuracy_score(labels, preds)

            if current_val_acc > highest_val_acc:
                highest_val_acc = current_val_acc
                best_trial_path = trial_path

        except FileNotFoundError:
            # print(f"Atlanıyor (Dosya bulunamadı): {trial_dir_name}")
            continue
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
        for model_name in MODELS:
            model_short_name = model_name.split('/')[-1]
            log_and_print("-" * 50, log_file)
            log_and_print(f"DEĞERLENDİRİLİYOR: {model_name}", log_file)
            log_and_print("-" * 50, log_file)

            # Güncellenmiş fonksiyon, train_v3.py'nin 'checkpoints_v3' klasörünü tarayacak
            best_trial_dir, best_val_acc = find_best_trial_by_rerunning_validation(model_name, BASE_CHECKPOINT_DIR,
                                                                                   device)

            if best_trial_dir is None:
                log_and_print(f"HATA: {model_name} için geçerli bir deneme (trial) bulunamadı. Bu model atlanıyor.",
                              log_file)
                continue

            log_and_print(
                f"En iyi deneme (trial) bulundu: {os.path.basename(best_trial_dir)} (Validation Accuracy: {best_val_acc:.4f})",
                log_file)
            log_and_print("Model ve bileşenler yükleniyor...", log_file)

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
            multiclass_model = TransformerClassifier(model_name=model_name, num_labels=len(multiclass_encoder.classes_))
            multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
            multiclass_model.load_state_dict(torch.load(multiclass_model_path, map_location=device))
            multiclass_model.to(device)

            log_and_print("Orijinal test veri seti (Context-Aware) yeniden oluşturuluyor...", log_file)
            # train_v3.py ile aynı veri setini ve ayarları kullan
            full_dataset = CitationDataset(tokenizer=tokenizer, csv_path=DATA_PATH, max_len=256, task=None,
                                           include_section_in_input=True)
            full_label_encoder = full_dataset.label_encoder
            label_names = full_label_encoder.classes_.tolist()
            generator = Generator().manual_seed(SEED)

            # 80% Train/Val
            train_val_size = int(0.8 * len(full_dataset))
            # Sadece Test setini (ikinci parçayı) al
            _, test_dataset = random_split(full_dataset, [train_val_size, len(full_dataset) - train_val_size],
                                           generator=generator)

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            log_and_print(f"Test için {len(test_dataset)} örnek yüklendi.", log_file)

            log_and_print("Hiyerarşik (Context-Aware) test süreci başlatılıyor...", log_file)
            predictions, true_labels = get_hierarchical_predictions(
                binary_model, multiclass_model, test_loader, device,
                binary_encoder, multiclass_encoder, full_label_encoder
            )

            accuracy = accuracy_score(true_labels, predictions)
            macro_f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)
            report = classification_report(true_labels, predictions, target_names=label_names, zero_division=0)
            log_and_print("\n--- TEST SONUÇLARI (Experiment 3 - Context-Aware Hierarchical) ---", log_file)
            log_and_print(f"Accuracy: {accuracy:.4f}", log_file)
            log_and_print(f"Macro F1-Score: {macro_f1:.4f}", log_file)
            log_and_print("Sınıflandırma Raporu:", log_file)
            log_and_print(report, log_file)

            cm = confusion_matrix(true_labels, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
            plt.title(f'Confusion Matrix (Experiment 3) - {model_short_name}', fontsize=16)
            plt.ylabel('Gerçek Etiket (True Label)', fontsize=12)
            plt.xlabel('Tahmin Edilen Etiket (Predicted Label)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_plot_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_short_name}.png")
            plt.savefig(cm_plot_path)
            plt.close()
            log_and_print(f"\nConfusion matrix kaydedildi: {cm_plot_path}", log_file)

            results.append({"model_name": model_short_name, "accuracy": accuracy, "macro_f1": macro_f1})

        summary_df = pd.DataFrame(results)
        summary_string = summary_df.to_string(index=False)
        log_and_print("\n\n" + "=" * 60, log_file)
        log_and_print("     TÜM MODELLERİN ÖZET SONUÇLARI (Experiment 3 - Context-Aware Hierarchical)", log_file)
        log_and_print("=" * 60, log_file)
        log_and_print(summary_string, log_file)
        log_and_print("=" * 60, log_file)

        print("\nSonuçlar görselleştiriliyor ve kaydediliyor...")
        plt.style.use('seaborn-v0_8-whitegrid')
        df_melted = summary_df.melt(id_vars='model_name', var_name='Metric', value_name='Score')
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df_melted, x='model_name', y='Score', hue='Metric', ax=ax, palette="viridis")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                        va='center', xytext=(0, 9), textcoords='offset points')
        ax.set_title('Experiment 3: Context-Aware Hierarchical Model Karşılaştırması', fontsize=16, fontweight='bold')
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