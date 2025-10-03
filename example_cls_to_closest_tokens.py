import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

# Model ve tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Test cümlesi
sentence = "Bu çalışma, yapay zeka tekniklerini kullanmaktadır."
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# [CLS] embedding'ini çıkar
with torch.no_grad():
    outputs = model(**inputs)
    cls_vector = outputs.pooler_output.squeeze(0)  # shape: [768]

# BERT'in tüm token embedding'leri
embedding_matrix = model.embeddings.word_embeddings.weight.data  # shape: [vocab_size, 768]

# Cosine benzerliğini hesapla
similarities = cosine_similarity(cls_vector.unsqueeze(0), embedding_matrix).squeeze()
topk = torch.topk(similarities, k=5)

# En yakın token'ları al
closest_token_ids = topk.indices.tolist()
closest_tokens = tokenizer.convert_ids_to_tokens(closest_token_ids)

print("\n[CLS] vektörüne en yakın token'lar:")
for i, token in enumerate(closest_tokens):
    print(f"{i+1}. {token} (benzerlik: {topk.values[i]:.4f})")
