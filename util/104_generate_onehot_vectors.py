import argparse
import os
import sys
import pickle
import pandas as pd
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Proje modülleri
from config import Config
from dataset import CitationDataset
from generic_model import TransformerClassifier

# ==============================================================================
#                      SABİT AYARLAR
# ==============================================================================

TARGET_ORDER = ['background', 'basis', 'support', 'differ', 'discuss']


# ==============================================================================
#                      YARDIMCI FONKSİYONLAR
# ==============================================================================

def get_flat_predictions(model, data_loader, device):
    """Experiment 1 (Flat) için tahmin fonksiyonu."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_args = {"input_ids": input_ids, "attention_mask": attention_mask}
            if batch.get("token_type_ids") is not None:
                model_args["token_type_ids"] = batch.get("token_type_ids").to(device)

            logits = model(**model_args)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds


def get_hierarchical_predictions(binary_model, multiclass_model, data_loader, device,
                                 binary_encoder, multiclass_encoder, full_label_encoder):
    """Experiment 2 ve 3 (Hiyerarşik) için tahmin fonksiyonu."""
    binary_model.eval()
    multiclass_model.eval()
    all_preds = []

    try:
        non_bg_id = binary_encoder.transform(['non-background'])[0]
        bg_orig_id = full_label_encoder.transform(['background'])[0]
    except Exception as e:
        print(f"Encoder hatası: {e}")
        return []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model_args = {"input_ids": input_ids, "attention_mask": attention_mask}
            if batch.get("token_type_ids") is not None:
                model_args["token_type_ids"] = batch.get("token_type_ids").to(device)

            # 1. Binary
            bin_logits = binary_model(**model_args)
            bin_preds = torch.argmax(bin_logits, dim=1)
            final_preds = torch.full_like(bin_preds, fill_value=-1)

            # 2. Expert
            expert_indices = (bin_preds == non_bg_id).nonzero(as_tuple=True)[0]
            if len(expert_indices) > 0:
                exp_input = input_ids[expert_indices]
                exp_mask = attention_mask[expert_indices]
                expert_args = {"input_ids": exp_input, "attention_mask": exp_mask}
                if "token_type_ids" in model_args:
                    expert_args["token_type_ids"] = model_args["token_type_ids"][expert_indices]

                multi_logits = multiclass_model(**expert_args)
                multi_preds_raw = torch.argmax(multi_logits, dim=1)

                cls_names = multiclass_encoder.inverse_transform(multi_preds_raw.cpu().numpy())
                global_ids = full_label_encoder.transform(cls_names)
                final_preds[expert_indices] = torch.tensor(global_ids, device=device)

            # 3. Background
            bg_indices = (bin_preds != non_bg_id).nonzero(as_tuple=True)[0]
            final_preds[bg_indices] = bg_orig_id

            all_preds.extend(final_preds.cpu().numpy())
    return all_preds


def find_best_trial(model_name, checkpoint_dir):
    model_short = Config.get_model_short_name(model_name)
    possible_dbs = [f"{model_short}.db", f"{model_short}_hierarchical.db", f"{model_short}_refined.db"]

    found_db = None
    for db_name in possible_dbs:
        db_path = os.path.join(checkpoint_dir, db_name)
        if os.path.exists(db_path):
            found_db = db_path
            break

    if found_db:
        try:
            storage_url = f"sqlite:///{found_db}"
            summaries = optuna.study.get_all_study_summaries(storage=storage_url)
            if summaries:
                study = optuna.load_study(study_name=summaries[0].study_name, storage=storage_url)
                trial_num = study.best_trial.number
                trial_path = os.path.join(checkpoint_dir, model_short, f"trial_{trial_num}")
                if os.path.exists(trial_path):
                    return trial_path
        except:
            pass

    # Manuel Tarama
    model_dir = os.path.join(checkpoint_dir, model_short)
    if os.path.exists(model_dir):
        trials = [d for d in os.listdir(model_dir) if d.startswith("trial_")]
        for trial in sorted(trials, reverse=True):
            t_path = os.path.join(model_dir, trial)
            if os.path.exists(os.path.join(t_path, "best_model.pt")) or \
                    os.path.exists(os.path.join(t_path, "binary/best_model.pt")):
                return t_path
    return None


def create_one_hot(label_name, target_order):
    vec = np.zeros(len(target_order), dtype=int)
    try:
        idx = target_order.index(label_name)
        vec[idx] = 1
    except ValueError:
        pass
    return vec


# ==============================================================================
#                      PROCESS DATASET
# ==============================================================================

def process_dataset_split(split_name, experiment_ids, model_indices, device, output_suffix):
    """
    Belirli bir split (train, val veya test) için tüm modelleri çalıştırır ve CSV kaydeder.
    """
    print(f"\n{'#' * 60}")
    print(f"BAŞLATILIYOR: {split_name.upper()} SETİ")
    print(f"{'#' * 60}")

    # 1. Ana İskelet Verisini Yükle (Referans olarak Exp 1 kullanılır)
    # Bu adımda sadece ID ve True Label'ları almak için temel veri setini yüklüyoruz.
    Config.set_experiment(1)

    if split_name == "train":
        data_path = Config.DATA_PATH_TRAIN
    elif split_name == "val":
        data_path = Config.DATA_PATH_VAL
    else:
        data_path = Config.DATA_PATH_TEST

    if not os.path.exists(data_path):
        print(f"HATA: {split_name} veri dosyası bulunamadı: {data_path}")
        return

    base_df = pd.read_csv(data_path)
    print(f"Referans Veri Yüklendi ({split_name}): {len(base_df)} satır.")

    # Sonuç DataFrame'i (ID ve True Label ile başla)
    if 'id' in base_df.columns:
        result_df = base_df[['id', 'citation_intent']].copy()
        result_df.rename(columns={'id': 'citation_id', 'citation_intent': 'true_label'}, inplace=True)
    else:
        result_df = base_df[['citation_intent']].copy()
        result_df.rename(columns={'citation_intent': 'true_label'}, inplace=True)
        result_df['citation_id'] = base_df.index

    # 2. Modelleri Döngüye Al
    for exp_id in experiment_ids:
        for model_idx in model_indices:

            # Config ve Path Ayarları
            Config.set_experiment(exp_id)
            Config.set_model(model_idx)

            # Bu experiment için doğru path'i belirle (örn: _ext uzantılı olabilir)
            if split_name == "train":
                current_data_path = Config.DATA_PATH_TRAIN
            elif split_name == "val":
                current_data_path = Config.DATA_PATH_VAL
            else:
                current_data_path = Config.DATA_PATH_TEST

            # DataFrame'i yeniden yükle (Context farkları için)
            current_df = pd.read_csv(current_data_path)

            model_name_full = Config.MODELS[model_idx]
            model_short = Config.get_model_short_name(model_name_full)
            model_prefix = f"{model_short}_{exp_id}"

            print(f"   >>> İşleniyor: {model_prefix}")

            best_trial_path = find_best_trial(model_name_full, Config.CHECKPOINT_DIR)
            if not best_trial_path:
                print(f"      UYARI: Checkpoint yok. Sütunlar 0 ile doldurulacak.")
                for label in TARGET_ORDER:
                    result_df[f"{model_prefix}_{label}"] = 0
                continue

            try:
                tokenizer = AutoTokenizer.from_pretrained(best_trial_path)
                predictions_indices = []
                current_label_encoder = None

                # --- MODEL YÜKLEME VE TAHMİN ---
                if Config.EXPERIMENT_ID in [1, 4]:  # FLAT
                    enc_path = os.path.join(best_trial_path, "label_encoder.pkl")
                    if not os.path.exists(enc_path): enc_path = os.path.join(best_trial_path, "label_encoder_flat.pkl")

                    with open(enc_path, "rb") as f:
                        current_label_encoder = pickle.load(f)

                    model = TransformerClassifier(model_name_full, num_labels=len(current_label_encoder.classes_))
                    model.transformer.resize_token_embeddings(len(tokenizer))
                    model.load_state_dict(
                        torch.load(os.path.join(best_trial_path, "best_model.pt"), map_location=device))
                    model.to(device)

                    ds = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                         data_frame=current_df, task=None, include_section_in_input=Config.CONTEXT_RICH)
                    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False)
                    predictions_indices = get_flat_predictions(model, loader, device)

                else:  # HIERARCHICAL
                    bin_dir = os.path.join(best_trial_path, "binary")
                    multi_dir = os.path.join(best_trial_path, "multiclass")

                    if not os.path.exists(bin_dir):
                        print("      HATA: Klasör yapısı bozuk.")
                        continue

                    with open(os.path.join(bin_dir, "label_encoder.pkl"), "rb") as f:
                        bin_enc = pickle.load(f)
                    with open(os.path.join(multi_dir, "label_encoder.pkl"), "rb") as f:
                        multi_enc = pickle.load(f)

                    ds = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled",
                                         data_frame=current_df, task="all",
                                         include_section_in_input=Config.CONTEXT_RICH)
                    current_label_encoder = ds.label_encoder

                    bin_model = TransformerClassifier(model_name_full, num_labels=len(bin_enc.classes_))
                    bin_model.transformer.resize_token_embeddings(len(tokenizer))
                    bin_model.load_state_dict(torch.load(os.path.join(bin_dir, "best_model.pt"), map_location=device))
                    bin_model.to(device)

                    multi_model = TransformerClassifier(model_name_full, num_labels=len(multi_enc.classes_))
                    multi_model.transformer.resize_token_embeddings(len(tokenizer))
                    multi_model.load_state_dict(
                        torch.load(os.path.join(multi_dir, "best_model.pt"), map_location=device))
                    multi_model.to(device)

                    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False)
                    predictions_indices = get_hierarchical_predictions(
                        bin_model, multi_model, loader, device, bin_enc, multi_enc, current_label_encoder
                    )

                # --- SONUÇLARI EKLE ---
                model_vectors = {label: [] for label in TARGET_ORDER}
                for pred_idx in predictions_indices:
                    pred_label_name = current_label_encoder.inverse_transform([pred_idx])[0]
                    one_hot = create_one_hot(pred_label_name.lower(), TARGET_ORDER)
                    for i, label in enumerate(TARGET_ORDER):
                        model_vectors[label].append(one_hot[i])

                for label in TARGET_ORDER:
                    result_df[f"{model_prefix}_{label}"] = model_vectors[label]

            except Exception as e:
                print(f"      HATA: {e}")
                # Hata durumunda 0 bas
                for label in TARGET_ORDER:
                    result_df[f"{model_prefix}_{label}"] = 0

    # 3. CSV Kaydet
    output_filename = f"data_v2_{split_name}_one_hot_{output_suffix}.csv"
    # Direkt çalışılan dizine kaydet
    result_df.to_csv(output_filename, index=False)
    print(f"✅ KAYDEDİLDİ: {output_filename}")


# ==============================================================================
#                      ANA AKIŞ
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate One-Hot Vectors for Train/Val/Test")
    parser.add_argument("--experiment_ids", type=int, nargs='+', required=False,
                        help="List of Experiment IDs (e.g. 1 3)")
    parser.add_argument("--model_indices", type=int, nargs='+', required=False, help="List of Model Indices (e.g. 0 2)")
    parser.add_argument("--file_suffix", type=str, default="104", help="Suffix for output files (e.g. '104')")
    args = parser.parse_args()

    # Default values
    args.experiment_ids = [1,2]
    args.model_indices = [0,1,2,3]

    # Cihaz
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Cihaz: {device}")
    print(f"İşlenecek Modeller: {args.model_indices}")
    print(f"İşlenecek Deneyler: {args.experiment_ids}")
    print(f"Çıktı Soneki: {args.file_suffix}")

    # 3 Veri seti için döngü
    splits = ["train", "val", "test"]

    for split in splits:
        process_dataset_split(split, args.experiment_ids, args.model_indices, device, args.file_suffix)

    print("\n✅ TÜM İŞLEMLER TAMAMLANDI.")


if __name__ == "__main__":
    main()