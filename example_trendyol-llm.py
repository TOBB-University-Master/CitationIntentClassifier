
"""
    !!!!! !!!!!
    TODO: Neden çalışmadığı araştırılacak...
    !!!!! !!!!!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modelin adını belirtiyoruz. Bu sefer çok daha büyük bir model!
model_name = "Trendyol/Trendyol-LLM-7b-chat-v1.0"

# 1. Tokenizer'ı Yükleme
# Tokenizer yine küçüktür ve hızlıca iner.
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Modeli Yükleme
# BU ADIM UZUN SÜREBİLİR VE ÇOK FAZLA RAM/VRAM KULLANIR!
# torch_dtype=torch.bfloat16, modeli daha az bellekte tutmak için bir optimizasyondur.
print(f"'{model_name}' modeli yükleniyor... Bu işlem birkaç dakika sürebilir.")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Modeli otomatik olarak uygun cihaza (GPU/CPU) yükler
)
print("Model başarıyla yüklendi!")

# 3. Pipeline ile Modeli Kullanma
# Üretken modeller için en kolay kullanım yolu pipeline'dır.
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Modele bir soru soralım (prompt)
prompt = "Türkiye'nin en güzel üç şehrini ve nedenlerini anlatır mısın?"

# Pipeline'ı kullanarak cevap üretelim
# max_new_tokens: Modelin en fazla kaç yeni token üreteceğini belirler.
# temperature: Cevabın ne kadar "yaratıcı" olacağını ayarlar (düşük değerler daha tutarlı).
print("\nCevap üretiliyor...")
result = generator(
    prompt,
    max_new_tokens=250,
    temperature=0.7
)

# Üretilen cevabı yazdır
print("\n--- Modelin Cevabı ---")
print(result[0]['generated_text'])