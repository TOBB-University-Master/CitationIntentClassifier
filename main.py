
from transformers import AutoTokenizer
from dataset import CitationDataset

"""
    *** NOTE ***
    Burada data bilgileri gösterilmektedir 
"""

# Eğitim script'inde kullandığınız model adını buraya yazın
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

print("Tokenizer yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Veri seti yükleniyor...")
# Veri setini oluşturuyoruz (CSV'den veya DB'den yükleyecek)
dataset = CitationDataset(tokenizer=tokenizer, csv_path="data/train.csv")

# İncelemek istediğimiz örneğin indeksini seçelim (örn: ilk örnek için 0)
sample_index = 0

print("\n" + "="*50)
print(f"ÖRNEK {sample_index} İNCELENİYOR")
print("="*50)

# 1. Adım: Verinin Orijinal Hali (DataFrame'den)
#----------------------------------------------------
original_row = dataset.df.iloc[sample_index]
print("\n[1] VERİNİN ORİJİNAL HALİ (CSV/DB'den geldiği gibi)")
print("----------------------------------------------------")
print(f"Orijinal Metin (citation_context): {original_row['citation_context']}")
print(f"Orijinal Bölüm (section): {original_row['section']}")
print(f"Orijinal Niyet (citation_intent): {original_row['citation_intent']}")


# 2. Adım: __getitem__ ile işlenmiş veri
#----------------------------------------------------
# Bu satır, dataset.py'deki __getitem__ metodunu çalıştırır
processed_sample = dataset[sample_index]
print("\n\n[2] VERİNİN İŞLENMİŞ HALİ (Modele Gidecek Tensörler)")
print("----------------------------------------------------")
# print(processed_sample) # Tüm sözlüğü görmek isterseniz bu satırı açabilirsiniz

input_ids = processed_sample["input_ids"]
attention_mask = processed_sample["attention_mask"]
label_id = processed_sample["label"]
section_id = processed_sample["section_id"]

print(f"input_ids (ilk 20 token): {input_ids[:20]}...")
print(f"attention_mask (ilk 20 token): {attention_mask[:20]}...")
print(f"label_id: {label_id.item()}") # .item() ile tensörden sayıyı alırız
print(f"section_id: {section_id.item()}")


# 3. Adım: Encoding İşlemini Anlaşılır Kılma
#----------------------------------------------------
print("\n\n[3] ENCODING DETAYLARI")
print("----------------------------------------------------")

# input_ids'yi tekrar metin token'larına çevirelim
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print(f"Toplam Token Sayısı: {len(tokens)}")
print(f"Gerçek Token Sayısı (Attention Mask = 1 olanlar): {attention_mask.sum().item()}")

# Padding token'larını sayalım
num_padding_tokens = len(tokens) - attention_mask.sum().item()
print(f"Padding Token Sayısı ([PAD]): {num_padding_tokens}")

print("\nToken'laştırılmış Metin (ilk 30 token):")
print(tokens[:30])

# Label ve Section ID'lerinin metin karşılıklarını bulalım
label_name = dataset.get_label_names()[label_id.item()]
section_name = dataset.get_section_names()[section_id.item()]
print("\nID -> Metin Dönüşümü:")
print(f"Label ID: {label_id.item()} -> Niyet Adı: '{label_name}'")
print(f"Section ID: {section_id.item()} -> Bölüm Adı: '{section_name}'")