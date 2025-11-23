import torch
import torch.nn as nn
import os
import logging
import pickle
import json
import optuna

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
from dataset import CitationDataset
from generic_model import TransformerClassifier
from tqdm import tqdm
from torch.amp import GradScaler, autocast

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI (STABİL) ***
# ==============================================================================
# VRAM dostu, test edilmiş bir modelle başlayalım
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
# MODEL_NAME = "dbmdz/electra-base-turkish-cased-discriminator"
# MODEL_NAME = "xlm-roberta-base"
# MODEL_NAME = "microsoft/deberta-v3-base"

DATA_PATH = "data/data_v2.csv"
# Veri setini incelemek için True yapıp çalıştırın
DATASET_INFO = False
# ==============================================================================


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Mevcut log handler'larını temizle (Optuna'nın tekrar tekrar handler eklemesini önler)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )


def evaluate(model, data_loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label_names, zero_division=0, output_dict=True)
    return acc, report


def run_training_stage(config, trial, task_type):
    """
    Belirtilen görev için (binary veya multiclass) bir eğitim aşamasını çalıştırır.
    (Stabilite için AMP ve doğru sıralama ile güncellenmiştir.)
    """
    is_binary = task_type == 'binary'
    task_name = "İkili" if is_binary else "Çok Sınıflı"

    output_dir = config["checkpoint_path_binary"] if is_binary else config["checkpoint_path_multiclass"]
    best_model_path = config["best_model_path_binary"] if is_binary else config["best_model_path_multiclass"]
    lr = config["lr_binary"] if is_binary else config["lr_multiclass"]
    epochs = config["epochs_binary"] if is_binary else config["epochs_multiclass"]

    log_file = os.path.join(output_dir, f"training_{task_type}.log")
    setup_logging(log_file)
    logging.info(f"--- Deneme #{trial.number} - {task_name} Eğitimi Başlatılıyor ---")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'], task=task_type)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()

    with open(config["label_encoder_binary_path"] if is_binary else config["label_encoder_multiclass_path"], "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_val_size
    train_val_dataset, _ = random_split(full_dataset, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = TransformerClassifier(model_name=config["model_name"], num_labels=num_labels)
    model.transformer.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    start_epoch, best_val_acc = 0, 0.0

    scaler = GradScaler()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{epochs} ({task_name})")

        for batch in progress_bar:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=torch.float16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_loss += loss.item()

        val_acc, _ = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"Epoch {epoch + 1} - {task_name} Doğrulama Başarımı: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi {task_name} model kaydediliyor: {best_val_acc:.4f}")
            torch.save(model.state_dict(), best_model_path)

    logging.info(f"--- {task_name} Sınıflandırıcı Eğitimi Tamamlandı ---")


def evaluate_hierarchical(config):
    logging.info("\n--- Birleşik Hiyerarşik Değerlendirme Başlatılıyor ---")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    full_dataset_orig = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=config['data_path'], task='all')

    with open(config["label_encoder_binary_path"], "rb") as f:
        binary_encoder = pickle.load(f)
    with open(config["label_encoder_multiclass_path"], "rb") as f:
        multiclass_encoder = pickle.load(f)

    non_background_binary_id = list(binary_encoder.transform(['non-background']))[0]

    binary_model = TransformerClassifier(model_name=config["model_name"], num_labels=len(binary_encoder.classes_))
    binary_model.transformer.resize_token_embeddings(len(tokenizer))
    binary_model.load_state_dict(torch.load(config["best_model_path_binary"], map_location=device))
    binary_model.to(device)
    binary_model.eval()

    multiclass_model = TransformerClassifier(model_name=config["model_name"],
                                             num_labels=len(multiclass_encoder.classes_))
    multiclass_model.transformer.resize_token_embeddings(len(tokenizer))
    multiclass_model.load_state_dict(torch.load(config["best_model_path_multiclass"], map_location=device))
    multiclass_model.to(device)
    multiclass_model.eval()

    generator = Generator().manual_seed(config["seed"])
    train_val_size = int(0.8 * len(full_dataset_orig))
    test_size = len(full_dataset_orig) - train_val_size
    train_val_dataset, _ = random_split(full_dataset_orig, [train_val_size, test_size], generator=generator)
    train_size = int(0.85 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    _, val_dataset_orig = random_split(train_val_dataset, [train_size, val_size], generator=generator)
    val_loader_orig = DataLoader(val_dataset_orig, batch_size=config["batch_size"])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader_orig:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)
            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)
            final_preds = torch.zeros_like(binary_preds)

            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]
            if len(expert_indices) > 0:
                expert_input_ids, expert_attention_mask = input_ids[expert_indices], attention_mask[expert_indices]
                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)
                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = full_dataset_orig.label_encoder.transform(multiclass_class_names)
                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            background_orig_id = full_dataset_orig.label_encoder.transform(['Background'])[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı: {overall_accuracy:.4f}")
    return overall_accuracy


def objective(trial):
    model_short_name = MODEL_NAME.split('/')[-1]
    output_dir_base = f"checkpoints_v2/{model_short_name}/trial_{trial.number}/"

    config = {
        "data_path": DATA_PATH, "model_name": MODEL_NAME, "seed": 42,
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        "lr_binary": trial.suggest_float("lr_binary", 1e-5, 5e-5, log=True),
        "lr_multiclass": trial.suggest_float("lr_multiclass", 1e-5, 5e-5, log=True),
        "epochs_binary": trial.suggest_int("epochs_binary", 2, 5),
        "epochs_multiclass": trial.suggest_int("epochs_multiclass", 5, 15),
        "checkpoint_path_binary": os.path.join(output_dir_base, "binary/"),
        "best_model_path_binary": os.path.join(output_dir_base, "binary/best_model.pt"),
        "label_encoder_binary_path": os.path.join(output_dir_base, "binary/label_encoder.pkl"),
        "checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir_base, "multiclass/best_model.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir_base, "multiclass/label_encoder.pkl"),
    }
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    run_training_stage(config, trial, 'binary')
    run_training_stage(config, trial, 'multiclass')
    overall_accuracy = evaluate_hierarchical(config)
    return overall_accuracy


def print_dataset_info(model_name, data_path, seed):
    print("--- Veri Seti Dağılım İncelemesi Başlatılıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def log_class_distribution(subset, name):
        labels = [subset[i]['label'].item() for i in range(len(subset))]
        counts = Counter(labels)
        original_dataset = subset
        while isinstance(original_dataset, Subset):
            original_dataset = original_dataset.dataset
        label_names = original_dataset.label_encoder.classes_
        print(f"\n--- {name} Sınıf Dağılımı ---")
        print(f"Toplam Örnek: {len(subset)}")
        for label_id, count in sorted(counts.items()):
            print(f"    {label_names[label_id]} (ID: {label_id}): {count}")

    for task in ['binary', 'multiclass']:
        print(f"\n{'=' * 20} GÖREV: {task.upper()} {'=' * 20}")
        full_dataset = CitationDataset(tokenizer=tokenizer, mode="labeled", csv_path=data_path, task=task)
        generator = Generator().manual_seed(seed)
        train_val_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=generator)
        train_size = int(0.85 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)
        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")


if __name__ == "__main__":
    if DATASET_INFO:
        print_dataset_info(model_name=MODEL_NAME, data_path=DATA_PATH, seed=42)
    else:
        model_short_name = MODEL_NAME.split('/')[-1]
        study_name = f"{model_short_name}_hierarchical_study"
        storage_path = f"sqlite:///{model_short_name}_hierarchical.db"

        print(f"🚀 Hiyerarşik Optimizasyon Başlatılıyor 🚀")
        print(f"Model: {MODEL_NAME}")
        print(f"Çalışma Adı (Study Name): {study_name}")
        print(f"Veritabanı Dosyası: {storage_path}")
        print("-------------------------------------------------")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True,
            direction="maximize"
        )
        study.optimize(objective, n_trials=50)

        print("\nOptimizasyon tamamlandı.")
        print("En iyi deneme:")
        trial = study.best_trial
        print(f"  Değer (En Yüksek Birleşik Doğrulum): {trial.value}")
        print("  En İyi Parametreler: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")