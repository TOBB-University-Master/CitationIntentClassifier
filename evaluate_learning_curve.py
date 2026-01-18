import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# Proje modülleri
from config import Config
from dataset import CitationDataset
from generic_model import TransformerClassifier

# ==============================================================================
# AYARLAR
# ==============================================================================
BASE_DIR = "_train_104"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"
MODEL_SHORT_NAME = MODEL_NAME.split('/')[-1]
TEST_DATA_PATH = "data/data_v2_test.csv"
FULL_TRAIN_DATA_PATH = "data/data_v2_train.csv"

# Karşılaştırılacak Modeller ve Klasörleri
# (Etiket, Klasör Yolu, Veri Seti Boyutu)
# Not: 'Full' boyutu otomatik hesaplanacak.
MODELS_CONFIG = [
    {"label": "500 Samples",  "folder": f"{BASE_DIR}/checkpoints_v1_learning_rate_500/{MODEL_SHORT_NAME}",  "size": 500},
    {"label": "1000 Samples", "folder": f"{BASE_DIR}/checkpoints_v1_learning_rate_1000/{MODEL_SHORT_NAME}", "size": 1000},
    {"label": "1500 Samples", "folder": f"{BASE_DIR}/checkpoints_v1_learning_rate_1500/{MODEL_SHORT_NAME}", "size": 1500},
    {"label": "Full Dataset", "folder": f"{BASE_DIR}/checkpoints_v1/{MODEL_SHORT_NAME}", "size": "FULL"}
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def get_full_dataset_size():
    """Full veri setinin satır sayısını döner."""
    if os.path.exists(FULL_TRAIN_DATA_PATH):
        df = pd.read_csv(FULL_TRAIN_DATA_PATH)
        return len(df)
    return 2000 # Varsayılan fallback

def find_best_model_path(folder_path):
    """Klasör altındaki best_model.pt dosyasını bulur."""
    # Önce trial klasörlerine bak
    search_pattern = os.path.join(folder_path, "**", "best_model.pt")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        return None
    
    # Birden fazla varsa, en son değiştirileni al (veya trial numarası en büyük olanı)
    # Burada en son değiştirileni alıyoruz.
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def find_label_encoder(folder_path):
    """Klasör altındaki label_encoder.pkl dosyasını bulur."""
    search_pattern = os.path.join(folder_path, "**", "label_encoder.pkl")
    files = glob.glob(search_pattern, recursive=True)
    if files:
        return max(files, key=os.path.getmtime)
    return None

def evaluate_model(model_path, encoder_path, test_loader, device):
    """Bir modeli test seti üzerinde değerlendirir."""
    print(f"   Model yükleniyor: {model_path}")
    
    # Encoder Yükle
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    
    num_labels = len(label_encoder.classes_)
    
    # Model Yükle
    model = TransformerClassifier(MODEL_NAME, num_labels=num_labels)
    # Tokenizer boyutunu ayarla (CITE token için)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})
    model.transformer.resize_token_embeddings(len(tokenizer))
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return acc, f1

def plot_learning_curve(results):
    """Sonuçları grafikleştirir."""
    df = pd.DataFrame(results)
    df = df.sort_values(by="size")
    
    print("\n--- SONUÇ TABLOSU ---")
    print(df)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Accuracy Line
    sns.lineplot(data=df, x="size", y="accuracy", marker='o', label='Accuracy', linewidth=2.5, color='tab:blue')
    # F1 Line
    sns.lineplot(data=df, x="size", y="f1_score", marker='s', label='Macro F1', linewidth=2.5, color='tab:orange', linestyle='--')
    
    plt.title(f"Learning Curve Analysis: {MODEL_SHORT_NAME}", fontsize=14)
    plt.xlabel("Training Set Size (Number of Samples)", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend()
    
    # Değerleri noktaların üzerine yaz
    for i in range(len(df)):
        plt.text(df["size"].iloc[i], df["accuracy"].iloc[i] + 0.01, f"{df['accuracy'].iloc[i]:.3f}", 
                 ha='center', color='tab:blue', fontweight='bold')
        plt.text(df["size"].iloc[i], df["f1_score"].iloc[i] - 0.03, f"{df['f1_score'].iloc[i]:.3f}", 
                 ha='center', color='tab:orange')

    output_file = "learning_curve_analysis.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nGrafik kaydedildi: {output_file}")

def main():
    print(f"--- Learning Curve Analizi Başlıyor ---")
    print(f"Cihaz: {DEVICE}")
    
    # 1. Full Dataset Boyutunu Belirle
    full_size = get_full_dataset_size()
    print(f"Full Dataset Boyutu: {full_size}")
    
    # Config'deki 'FULL' değerini güncelle
    for cfg in MODELS_CONFIG:
        if cfg["size"] == "FULL":
            cfg["size"] = full_size
            
    # 2. Test Verisini Hazırla (Sadece bir kere)
    print("Test verisi yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<CITE>']})
    
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_dataset = CitationDataset(tokenizer, max_len=Config.MAX_LEN, mode="labeled", data_frame=test_df)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)
    
    results = []
    
    # 3. Modelleri Değerlendir
    for cfg in MODELS_CONFIG:
        print(f"\nDeğerlendiriliyor: {cfg['label']} (Size: {cfg['size']})")
        
        model_path = find_best_model_path(cfg["folder"])
        encoder_path = find_label_encoder(cfg["folder"])
        
        if model_path and encoder_path:
            try:
                acc, f1 = evaluate_model(model_path, encoder_path, test_loader, DEVICE)
                print(f"   -> Accuracy: {acc:.4f} | F1: {f1:.4f}")
                
                results.append({
                    "size": cfg["size"],
                    "accuracy": acc,
                    "f1_score": f1,
                    "label": cfg["label"]
                })
            except Exception as e:
                print(f"   HATA: Model değerlendirilemedi. {e}")
        else:
            print(f"   UYARI: Model dosyası bulunamadı! ({cfg['folder']})")
            # Grafik bozulmasın diye boş geçiyoruz, ama isterseniz 0 ekleyebilirsiniz.
    
    if results:
        plot_learning_curve(results)
    else:
        print("\nHiçbir model değerlendirilemedi.")

if __name__ == "__main__":
    main()
