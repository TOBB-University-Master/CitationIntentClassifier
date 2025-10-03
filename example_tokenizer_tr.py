from transformers import AutoTokenizer

# train.py ve dataset.py'de kullandığımız aynı tokenizer'ı yüklüyoruz.
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset'inizde belirttiğiniz maksimum uzunluk
MAX_LEN = 128

# --- Örnek Cümleler ---
# 1. Normal, kısa bir cümle
cumle_1 = "Bu çalışmada derin öğrenme kullanıldı."

# 2. MAX_LEN'den (128 token) muhtemelen daha uzun bir cümle
cumle_2 = ("Bu çalışmanın temel amacı, doğal dil işleme tekniklerini kullanarak metin sınıflandırma "
           "problemlerine yeni bir yaklaşım sunmaktır. Özellikle, önceden eğitilmiş BERT "
           "modellerinin Türkçe metinler üzerindeki performansını artırmak için veri artırma "
           "yöntemleri ve hiperparametre optimizasyonu gibi çeşitli stratejiler incelenmiştir. "
           "Elde edilen sonuçlar, önerilen yöntemin mevcut en son teknolojiye sahip modellere "
           "kıyasla daha yüksek doğruluk oranları sunduğunu göstermektedir.")

# 3. Türkçe'nin ek yapısını ve "subword" tokenization'ı gösteren bir cümle
cumle_3 = "Veri setini Ankara'daki bilgisayarlarla işledik."


def print_details(sentence, encoding):
    """Tokenizer çıktısını detaylı bir şekilde yazdıran yardımcı fonksiyon."""
    print(f"\n--- İNCELENEN CÜMLE ---\n'{sentence}'")

    input_ids = encoding["input_ids"].squeeze()  # Squeeze ile [1, 128] -> [128] yapıyoruz
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
print("=" * 20, "ÖRNEK 1: KISA CÜMLE", "=" * 20)
encoding_1 = tokenizer(
    cumle_1,
    padding="max_length",  # 128'e kadar [PAD] ekle
    truncation=True,  # 128'den uzunsa kes
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(cumle_1, encoding_1)

# Örnek 2: Uzun Cümle
print("\n\n", "=" * 20, "ÖRNEK 2: UZUN CÜMLE", "=" * 20)
encoding_2 = tokenizer(
    cumle_2,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(cumle_2, encoding_2)

# Örnek 3: Subword Cümlesi
print("\n\n", "=" * 20, "ÖRNEK 3: SUBWORD TOKENIZATION", "=" * 20)
encoding_3 = tokenizer(
    cumle_3,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)
print_details(cumle_3, encoding_3)