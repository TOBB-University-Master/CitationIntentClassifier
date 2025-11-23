from transformers import AutoTokenizer

# En yaygın İngilizce BERT modellerinden birini yüklüyoruz.
# 'uncased' -> büyük-küçük harf ayrımı yapmaz, her şeyi küçük harfe çevirir.
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset'inizde belirttiğiniz maksimum uzunluk
MAX_LEN = 128

# --- Örnek İngilizce Cümleler ---
# 1. Normal, kısa bir cümle (Türkçe Örnek 1'in karşılığı)
sentence_1 = "Deep learning was used in this study."

# 2. MAX_LEN'den (128 token) daha uzun bir cümle (Türkçe Örnek 2'nin karşılığı)
sentence_2 = ("The primary objective of this research is to introduce a novel approach to text "
              "classification problems using advanced natural language processing techniques. Specifically, "
              "we investigate various strategies such as data augmentation and hyperparameter optimization "
              "to enhance the performance of pre-trained BERT models on English texts. The results "
              "obtained demonstrate that the proposed methodology offers superior accuracy rates "
              "compared to existing state-of-the-art models, paving the way for future developments.")

# 3. Subword tokenization'ı gösteren bir cümle (Türkçe Örnek 3'ün karşılığı)
sentence_3 = "We are exploring tokenization's complexities."


def print_details(sentence, encoding):
    """Tokenizer çıktısını detaylı bir şekilde yazdıran yardımcı fonksiyon."""
    print(f"\n--- İNCELENEN CÜMLE ---\n'{sentence}'")

    input_ids = encoding["input_ids"].squeeze()
    attention_mask = encoding["attention_mask"].squeeze()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print(f"\n[1] Token'lara Ayrılmış Hali (ilk 25 token):")
    print(tokens[:25])

    print(f"\n[2] Input IDs (Sayısal Karşılıklar - ilk 25):")
    print(input_ids.tolist()[:25])

    print(f"\n[3] Attention Mask (ilk 25):")
    print(attention_mask.tolist()[:25])

    print("\n--- ÖZET ---")
    print(f"Toplam Token Sayısı (Padding Dahil): {len(tokens)}")

    real_token_count = attention_mask.sum().item()
    print(f"Gerçek Token Sayısı (Padding Hariç): {real_token_count}")
    print(f"Padding Token Sayısı: {len(tokens) - real_token_count}")
    print("-" * 50)


# --- Örnekleri Çalıştıralım ---

# Örnek 1: Kısa Cümle
print("=" * 20, "ÖRNEK 1: KISA CÜMLE (İNGİLİZCE)", "=" * 20)
encoding_1 = tokenizer(
    sentence_1,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(sentence_1, encoding_1)

# Örnek 2: Uzun Cümle
print("\n\n", "=" * 20, "ÖRNEK 2: UZUN CÜMLE (İNGİLİZCE)", "=" * 20)
encoding_2 = tokenizer(
    sentence_2,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(sentence_2, encoding_2)

# Örnek 3: Subword Cümlesi
print("\n\n", "=" * 20, "ÖRNEK 3: SUBWORD TOKENIZATION (İNGİLİZCE)", "=" * 20)
encoding_3 = tokenizer(
    sentence_3,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(sentence_3, encoding_3)