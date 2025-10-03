import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import argparse

# Argparse: model ve parametre ayarları
parser = argparse.ArgumentParser(description="Embedding'e en yakın token'ları bul")
parser.add_argument("--embedding_type", type=str, default="cls", help="cls | mean | max | attn")
parser.add_argument("--timestamp", type=str, default="202505200833", help="Zaman etiketi (örn: 202505201434)")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/berturk_classifier_checkpoint.pt", help="Eğitilmiş model checkpoint yolu (opsiyonel)")
args = parser.parse_args()

# Embedding türünü buradan ayarlayabilirsin ("cls", "mean", "max", "attn")
embedding_type = args.embedding_type      # Embedding type
timestamp = args.timestamp                # Dosya adındaki timestamp

# Check Directories
os.makedirs("output", exist_ok=True)

# Model ve tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Eğitilmiş checkpoint varsa yükle
if args.checkpoint_path and os.path.exists(args.checkpoint_path):
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print("Eğitilmiş model yüklendi.")

model.eval()

# Veriyi yükle
embedding_path = f"output/{timestamp}_{embedding_type}_embedding.npy"
label_path = f"output/{timestamp}_labels.npy"

embeddings = np.load(embedding_path)
labels = np.load(label_path)

# Token embedding matrisini al
embedding_matrix = model.embeddings.word_embeddings.weight.data  # shape: [vocab_size, 768]

# Tüm cümle embedding'leri için en yakın 3 token
for i, emb in enumerate(embeddings):
    emb_tensor = torch.tensor(emb).unsqueeze(0)  # shape: [1, 768]
    sims = cosine_similarity(emb_tensor, embedding_matrix).squeeze()
    topk = torch.topk(sims, k=20)
    top_ids = topk.indices.tolist()
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)

    print(f"\nCümle {i+1} için en yakın token'lar ({embedding_type.upper()}):")
    for j, token in enumerate(top_tokens):
        print(f"  {j+1}. {token} (benzerlik: {topk.values[j]:.4f})")
