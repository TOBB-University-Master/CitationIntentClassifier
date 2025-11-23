import torch
from transformers import AutoModel, AutoTokenizer
from dataset import CitationDataset
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Dataset (etiketsiz mod)
db_url = "mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs"
dataset = CitationDataset( mode="unlabeled",csv_path="data/train.csv")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Extract embeddings işlemi için " + str(dataset.mode) + " verisi çekildi")

# Dosya adı için zaman etiketi
timestamp = datetime.now().strftime("%Y%m%d%H%M")

# Embedding türlerine göre ayrı ayrı listeler
embeddings_cls = []
embeddings_mean = []
embeddings_max = []
embeddings_attn = []
labels = []

print("Extract embeddings işlemi için " + str(dataset.mode) + " verisi çekildi")

"""
    Embedding extraction olduğundan backpropagation hesaplamasına gerek yok
    Bu sebepten torch.no_grad yani gradient hesaplama kapatılmıştır  
"""
with torch.no_grad():
    for i, batch in enumerate(loader):
        print(str(i+1) + " / " + str(len(loader.dataset)) + " → " + loader.dataset.texts[i])
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        """
             [CLS] = Classification Token
             [CLS] Bu bir örnek cümledir . [SEP]
        """
        # CLS embedding
        cls_embedding = outputs.pooler_output.squeeze().numpy()
        embeddings_cls.append(cls_embedding)

        # Mean pooling
        mean_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings_mean.append(mean_embedding)

        # Max pooling
        max_embedding = outputs.last_hidden_state.max(dim=1).values.squeeze().numpy()
        embeddings_max.append(max_embedding)

        # Attention-weighted pooling (son katmanın attention'ı)
        if outputs.attentions:
            attn = outputs.attentions[-1]  # son attention layer: [batch, heads, seq_len, seq_len]
            attn_weights = attn.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
            token_weights = attn_weights[0]  # [CLS] token'ının dikkat dağılımı
            token_weights = token_weights / token_weights.sum()  # normalize et
            weighted_embedding = (outputs.last_hidden_state.squeeze(0) * token_weights.unsqueeze(1)).sum(dim=0)
            embeddings_attn.append(weighted_embedding.numpy())
        else:
            embeddings_attn.append(np.zeros(model.config.hidden_size))

        labels.append(label.item())

# Kaydetme
directory = "output"
os.makedirs(directory, exist_ok=True)

np.save(f"{directory}/{timestamp}_cls_embedding.npy", np.array(embeddings_cls))
np.save(f"{directory}/{timestamp}_mean_embedding.npy", np.array(embeddings_mean))
np.save(f"{directory}/{timestamp}_max_embedding.npy", np.array(embeddings_max))
np.save(f"{directory}/{timestamp}_attn_embedding.npy", np.array(embeddings_attn))
np.save(f"{directory}/{timestamp}_labels.npy", np.array(labels))

print("Tüm embedding ve etiketler kaydedildi.")
