import torch
import torch.nn as nn
import os
import logging
import pickle
import json
import optuna
import argparse

from torch import Generator
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, random_split, Subset
from transformers import get_scheduler, AutoTokenizer
from torch.optim import AdamW
from collections import Counter
from dataset import CitationDataset
from generic_model import TransformerClassifier
from tqdm import tqdm
from functools import partial

# ==============================================================================
#                      *** DENEY YAPILANDIRMASI ***
# ==============================================================================
MODEL_NAMES = [
    "dbmdz/bert-base-turkish-cased",
    "dbmdz/electra-base-turkish-cased-discriminator",
    "xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "answerdotai/ModernBERT-base"
]

DATA_PATH = "data/data_v2.csv"
CHECKPOINT_DIR = "checkpoints_v2_01"

DATASET_INFO = False
NUMBER_TRIALS = 20
NUMBER_EPOCHS = 40
DEFAULT_MODEL_INDEX = 4
# ==============================================================================


"""
     Eğitim sürecindeki önemli bilgileri (epoch başlangıcı, kayıp değeri, doğruluk vb.) hem bir dosyaya (training.log) 
     hem de konsola yazdırmak için bir loglama sistemi kurar
"""


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
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

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=False
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    val_macro_f1 = report_dict['macro avg']['f1-score']

    return acc, report_str, val_macro_f1


def display_samples(loader_name, data_loader, tokenizer, num_samples=1000):
    """
    Verilen bir DataLoader'dan belirtilen sayıda örneği yazdırır.
    """
    print(f"\n--- {loader_name} İçin {num_samples} Örnek Veri ---")

    # DataLoader'ı bir iteratöre dönüştür
    data_iter = iter(data_loader)

    # Örnekleri al ve yazdır
    for i in range(num_samples):
        try:
            sample = next(data_iter)

            # batch_size > 1 olabileceğinden her zaman batch'in ilk örneğini alıyoruz
            input_ids = sample['input_ids'][0]
            label = sample['label'][0]

            # Token ID'lerini tekrar okunabilir metne dönüştür
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

            print(f"\nÖrnek #{i + 1}:")
            print(f"  Okunabilir Metin: '{decoded_text}'")
            print(f"  Atanmış Label ID: {label.item()}")

        except StopIteration:
            print(f"\nUyarı: '{loader_name}' içinde {num_samples} adetten az veri var.")
            break
    print("-" * (len(loader_name) + 25))


def run_training_stage(config, trial, task_type):
    """
    Belirtilen görev için (binary veya multiclass) bir eğitim aşamasını çalıştırır.
    """
    is_binary = task_type == 'binary'

    # Dinamik olarak doğru yolları ve parametreleri seç
    output_dir = config["checkpoint_path_binary"] if is_binary else config["checkpoint_path_multiclass"]
    best_model_path = config["best_model_path_binary"] if is_binary else config["best_model_path_multiclass"]
    resume_checkpoint_path = config["resume_checkpoint_path_binary"] if is_binary else config["resume_checkpoint_path_multiclass"]
    lr = config["lr_binary"] if is_binary else config["lr_multiclass"]
    epochs = config["epochs_binary"] if is_binary else config["epochs_multiclass"]
    encoder_path = config["label_encoder_binary_path"] if is_binary else config["label_encoder_multiclass_path"]

    log_file = os.path.join(output_dir, f"training_{task_type}.log")
    setup_logging(log_file)
    logging.info(f"--- Deneme #{trial.number} - {task_type} Sınıflandırıcı Eğitimi Başlatılıyor ---")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    full_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=config['data_path'],
                                   task=task_type)
    num_labels = len(full_dataset.get_label_names())
    label_names_list = full_dataset.get_label_names()

    with open(encoder_path, "wb") as f:
        pickle.dump(full_dataset.label_encoder, f)
    logging.info(f"{task_type.capitalize()} label encoder şuraya kaydedildi: {encoder_path}")

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


    start_epoch, best_val_f1 = 0, 0.0
    best_val_acc = 0.0
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} Epoch {epoch + 1}/{epochs} ({task_type})")
        for batch in progress_bar:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), \
            batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        # Değerlendirme 3 değer döndürüyor
        val_acc, val_report_str, val_macro_f1 = evaluate(model, val_loader, device, label_names_list)
        logging.info(f"Epoch {epoch + 1} - {task_type} Doğrulama Başarımı (Accuracy): {val_acc:.4f}")
        logging.info(f"Epoch {epoch + 1} - {task_type} Doğrulama Başarımı (Macro F1): {val_macro_f1:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info(f"🚀 Yeni en iyi accuracy {task_type} model (Accuracy: {best_val_acc:.4f}) ...")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            logging.info(f"🚀 Yeni en iyi F1 {task_type} model (Macro F1: {best_val_f1:.4f}) kaydediliyor...")
            torch.save(model.state_dict(), best_model_path)

    logging.info(f"--- {task_type} Sınıflandırıcı Eğitimi Tamamlandı ---")


def evaluate_hierarchical(config):
    """
    Eğitilmiş ikili ve uzman modellerle hiyerarşik birleşik performansı ölçer.
    """
    logging.info("\n--- Birleşik Hiyerarşik Değerlendirme Başlatılıyor ---")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Gerekli tüm bileşenleri yükle
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # ÖNEMLİ: Değerlendirme için tüm sınıfları içeren orijinal veri setini kullan
    full_dataset_orig = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=config['data_path'],task='all')

    # İkili ve Çok Sınıflı görevlerin label encoder'larını yükle
    with open(config["label_encoder_binary_path"], "rb") as f:
        binary_encoder = pickle.load(f)
    with open(config["label_encoder_multiclass_path"], "rb") as f:
        multiclass_encoder = pickle.load(f)

    # İkili modelin "Non-Background" etiketinin ID'sini bul
    non_background_binary_id = list(binary_encoder.transform(['non-background']))[0]

    # Modelleri oluştur ve eğitilmiş en iyi ağırlıkları yükle
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

    # Orijinal veri setinden doğrulama (validation) setini ayır
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

            # Adım 1: Üst seviye model ile tahmin yap
            binary_logits = binary_model(input_ids, attention_mask)
            binary_preds = torch.argmax(binary_logits, dim=1)

            final_preds = torch.zeros_like(binary_preds)

            # Adım 2: Uzman modele danışılacak verileri belirle
            expert_indices = (binary_preds == non_background_binary_id).nonzero(as_tuple=True)[0]

            if len(expert_indices) > 0:
                # Sadece uzmanlık gerektiren input'ları seç
                expert_input_ids = input_ids[expert_indices]
                expert_attention_mask = attention_mask[expert_indices]

                # Uzman model ile tahmin yap
                multiclass_logits = multiclass_model(expert_input_ids, expert_attention_mask)
                multiclass_preds_raw = torch.argmax(multiclass_logits, dim=1)

                # Uzman modelin tahminlerini (0,1,2,3) orijinal etiketlere dönüştür
                multiclass_class_names = multiclass_encoder.inverse_transform(multiclass_preds_raw.cpu().numpy())
                multiclass_preds_orig_ids = full_dataset_orig.label_encoder.transform(multiclass_class_names)

                final_preds[expert_indices] = torch.tensor(multiclass_preds_orig_ids, device=device)

            # İkili modelin "Background" dediği verilerin etiketini de ekle (ID: 0)
            background_indices = (binary_preds != non_background_binary_id).nonzero(as_tuple=True)[0]
            background_orig_id = full_dataset_orig.label_encoder.transform(['background'])[0]
            final_preds[background_indices] = background_orig_id

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_accuracy = accuracy_score(all_labels, all_preds)
    # Orijinal (tüm sınıflar) etiket isimlerini al
    orig_label_names = full_dataset_orig.get_label_names()

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=orig_label_names,
        zero_division=0,
        output_dict=True
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=orig_label_names,
        zero_division=0,
        output_dict=False
    )

    # Birleşik makro F1 skorunu al
    overall_macro_f1 = report_dict['macro avg']['f1-score']

    # Loglamayı güncelle
    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı (Accuracy): {overall_accuracy:.4f}")
    logging.info(f"🏆 Birleşik Hiyerarşik Doğrulama Başarımı (Macro F1): {overall_macro_f1:.4f}")
    logging.info(f"Birleşik Sınıflandırma Raporu:\n{report_str}")

    # Optuna'ya F1 skorunu döndür
    return overall_macro_f1


def objective(trial, model_name):
    model_short_name = model_name.split('/')[-1]
    output_dir_base = f"{CHECKPOINT_DIR}/{model_short_name}/trial_{trial.number}/"

    config = {
        "data_path": DATA_PATH,
        "model_name": model_name,
        "seed": 42,

        # Denenecek Hiperparametreler
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "lr_binary": trial.suggest_float("lr_binary", 1e-5, 5e-5, log=True),
        "lr_multiclass": trial.suggest_float("lr_multiclass", 1e-5, 5e-5, log=True),
        "epochs_binary": NUMBER_EPOCHS,
        "epochs_multiclass": NUMBER_EPOCHS,
        # "epochs_binary": trial.suggest_int("epochs_binary", 2, 8),
        # "epochs_multiclass": trial.suggest_int("epochs_multiclass", 5, 20),

        # Yollar
        "checkpoint_path_binary": os.path.join(output_dir_base, "binary/"),
        "best_model_path_binary": os.path.join(output_dir_base, "binary/best_model.pt"),
        "resume_checkpoint_path_binary": os.path.join(output_dir_base, "binary/training_checkpoint.pt"),
        "label_encoder_binary_path": os.path.join(output_dir_base, "binary/label_encoder.pkl"),

        "checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/"),
        "best_model_path_multiclass": os.path.join(output_dir_base, "multiclass/best_model.pt"),
        "resume_checkpoint_path_multiclass": os.path.join(output_dir_base, "multiclass/training_checkpoint.pt"),
        "label_encoder_multiclass_path": os.path.join(output_dir_base, "multiclass/label_encoder.pkl"),
    }
    os.makedirs(config["checkpoint_path_binary"], exist_ok=True)
    os.makedirs(config["checkpoint_path_multiclass"], exist_ok=True)

    # 1. Aşama: İkili Modeli Eğit
    run_training_stage(config, trial, 'binary')

    # 2. Aşama: Uzman Modeli Eğit
    run_training_stage(config, trial, 'multiclass')

    # 3. Aşama: İki modelin ortak performansını ölç
    overall_macro_f1 = evaluate_hierarchical(config)

    logging.info(f"Deneme #{trial.number} tamamlandı. Tokenizer ve yapılandırma kaydediliyor...")

    # Tokenizer'ı özel token ile birlikte kaydet
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    special_tokens_dict = {'additional_special_tokens': ['<CITE>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.save_pretrained(output_dir_base)  # Ana deneme klasörüne kaydet

    config_path = os.path.join(output_dir_base, "trial_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Optuna'ya optimize edeceği değeri döndür
    return overall_macro_f1


def print_dataset_info(model_name, data_path, seed):
    """
    Veri setlerini yükler, böler ve her bir bölümdeki sınıf dağılımlarını loglar.
    Eğitim yapmaz, sadece bilgi verir.
    """
    print("--- Veri Seti Dağılım İncelemesi Başlatılıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # log_class_distribution yardımcı fonksiyonunu train_v2.py'den alıyoruz
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

    # İkili ve Çok Sınıflı görevler için döngü
    for task in ['binary', 'multiclass']:
        print(f"\n{'=' * 20} GÖREV: {task.upper()} {'=' * 20}")

        # Veri setini ilgili görev için yükle
        full_dataset = CitationDataset(tokenizer=tokenizer, max_len=128, mode="labeled", csv_path=data_path, task=task)

        # Veriyi ayırma (Train/Val/Test)
        generator = Generator().manual_seed(seed)
        train_val_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(full_dataset, [train_val_size, test_size], generator=generator)

        train_size = int(0.85 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=generator)

        # Her bir set için dağılımı göster
        log_class_distribution(train_dataset, "Eğitim Seti")
        log_class_distribution(val_dataset, "Doğrulama Seti")
        log_class_distribution(test_dataset, "Test Seti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Classifier Training with Optuna")
    parser.add_argument("--model_index", type=int, default=DEFAULT_MODEL_INDEX, help="Index of the model to train from MODEL_NAMES list.")
    args = parser.parse_args()
    model_index = args.model_index

    if DATASET_INFO:
        print_dataset_info(model_name=MODEL_NAMES[0], data_path=DATA_PATH, seed=42)
    else:

        try:
            model_name = MODEL_NAMES[model_index]
            print(f"\n\n{'=' * 60}")
            print(f"--- BAŞLATILIYOR: {model_name} için {NUMBER_TRIALS} denemelik optimizasyon ---")
            print(f"{'=' * 60}\n")

            # Optuna çalışma dizini ve çalışma adı ayarları
            model_short_name = model_name.split('/')[-1]
            study_name = f"{model_short_name}_hiearchical_study"
            # DB dosyasının ana dizine değil, CHECKPOINT_DIR içine kaydedilmesi daha düzenli olabilir
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            storage_path = f"sqlite:///{CHECKPOINT_DIR}/{model_short_name}_hierarchical.db"

            print(f"🚀 Hiyerarşik Optimizasyon Başlatılıyor 🚀")
            print(f"Model: {model_name}")
            print(f"Çalışma Adı (Study Name): {study_name}")
            print(f"Veritabanı Dosyası: {storage_path}")
            print("-------------------------------------------------")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_path,
                load_if_exists=True,
                direction="maximize"
            )

            # Optuna'ya o anki model_name'i geçmek için `functools.partial` kullanıyoruz.
            objective_with_model = partial(objective, model_name=model_name)

            study.optimize(objective_with_model, n_trials=NUMBER_TRIALS)

            print("\nOptimizasyon tamamlandı.")
            print("En iyi deneme:")
            trial = study.best_trial
            print(f"  Değer (En Yüksek Birleşik Macro F1): {trial.value}")
            print("  En İyi Parametreler: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")

        except IndexError:
            print(f"HATA: Geçersiz model_index: {model_index}. Bu değer 0 ile {len(MODEL_NAMES) - 1} arasında olmalıdır.")
            exit(1)

        except Exception as e:
            print(f"KRİTİK HATA: {model_name} için optimizasyon durduruldu. Hata: {e}")

        print(f"\n\n{'=' * 60}")
        print(f"--- {model_name} OPTİMİZASYONU TAMAMLANDI ---")
        print(f"{'=' * 60}")