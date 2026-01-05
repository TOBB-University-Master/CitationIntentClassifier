from transformers import AutoModel
import torch.nn as nn
import torch

"""
Modelinizin mevcut yapısı (Mean Pooling + Tek Katmanlı Linear) aslında "zayıf" değil, standart ve sağlam bir başlangıç noktasıdır. 
Literatürde buna "Global Average Pooling" denir ve genellikle iyi çalışır.

Ancak, Linear sınıflandırıcının (tek bir katman) kapasitesi sınırlıdır. BERT'ten gelen 768 boyutlu vektörleri alır ve bunları doğrudan sınıflara ayırmaya çalışır. 
Eğer verinizdeki sınıflar arasındaki ayrım, bu 768 boyutlu uzayda düz bir çizgiyle (lineer olarak) ayrılamayacak kadar karmaşıksa, model zorlanabilir.

Modelinizin başarımını artırmak için iki güçlü alternatif yaklaşım uygulayabiliriz:

1. Classification Head'i Güçlendirmek (MLP Head)
Tek bir Linear katman yerine, araya bir "Hidden Layer" (Gizli Katman) ve bir Aktivasyon Fonksiyonu (Tanh veya ReLU) ekleyerek modelin karar verme yeteneğini (non-linearity) artırabiliriz.

Mevcut: BERT -> MeanPool -> Linear -> Output

Önerilen: BERT -> MeanPool -> Linear -> Tanh -> Dropout -> Linear -> Output

Bu yapı, modelin özellikler arasındaki daha karmaşık ilişkileri öğrenmesini sağlar.

2. Pooling Stratejisini Değiştirmek
Siz şu an sadece Mean Pooling (tüm kelimelerin ortalaması) kullanıyorsunuz. Bu bazen "keskin" bilgilerin kaybolmasına neden olabilir.

Öneri (Concatenation): [CLS] tokenı (cümlenin genel temsili) ile Mean Pooling'i birleştirmek. Bu sayede hem genel bağlamı hem de ortalama anlamı yakalarsınız.

Güncellenmiş ve Güçlendirilmiş Model Kodu
Aşağıdaki kod, generic_model.py dosyanızın yerini alabilecek, hem MLP yapısına sahip hem de Pooling stratejisi güçlendirilmiş (CLS + Mean) halidir.
"""

"""
Neden Bu Daha İyi Olabilir?
Daha Fazla Bilgi (Pooling): Sadece ortalamayı almak bazen "sivri" ve önemli kelimelerin etkisini sönümleyebilir. 
[CLS] tokenı modelin cümle hakkındaki kendi özetidir. İkisini birleştirmek (torch.cat) modele hem kendi özetini hem de kelimelerin ortalamasını kullanma şansı verir.

Daha İyi Karar (MLP): Araya eklediğimiz Linear -> Tanh -> Dropout katmanı, modelin veriyi "bükererek" daha iyi ayırmasını sağlar. 
Özellikle atıf niyeti (Citation Intent) gibi bağlamın ince nüanslara bağlı olduğu durumlarda bu ekstra derinlik işe yarar.

Öneri: Bu değişikliği yaparsanız modelinizin parametre sayısı çok az artar (ihmal edilebilir düzeyde) ancak öğrenme kapasitesi artar. 
train.py içinde herhangi bir değişiklik yapmanıza gerek yoktur, sadece generic_model.py içeriğini bununla değiştirmeniz yeterlidir.
"""

class TransformerClassifier(nn.Module):
    """
    Geliştirilmiş Transformer Sınıflandırıcı.
    Özellikler:
    1. CLS Token + Mean Pooling birleşimi (Daha zengin temsil).
    2. MLP (Multi-Layer Perceptron) Sınıflandırma Başlığı (Daha güçlü karar mekanizması).
    """

    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()

        self.transformer = AutoModel.from_pretrained(model_name)

        # Modelin gizli katman boyutunu al (Örn: BERT için 768)
        hidden_size = self.transformer.config.hidden_size

        # --- YENİLİK 1: Gelişmiş Sınıflandırma Başlığı (MLP Head) ---
        # Tek bir Linear yerine, araya bir katman daha ekliyoruz.
        # Giriş boyutu * 2 olmasının sebebi aşağıda CLS ve Mean'i birleştirecek olmamız.
        combined_hidden_size = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_hidden_size, hidden_size),  # Ara katman (Projection)
            nn.Tanh(),  # Non-linearity (Karmaşıklığı öğrenir)
            nn.Dropout(dropout_rate),  # Ezberlemeyi önler
            nn.Linear(hidden_size, num_labels)  # Son karar katmanı
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Argümanları hazırla
        model_args = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            model_args["token_type_ids"] = token_type_ids

        # Transformer çıktısı
        outputs = self.transformer(**model_args)

        # last_hidden_state -> (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # --- YENİLİK 2: Hibrit Pooling Stratejisi ---

        # A) Mean Pooling (Sizin mevcut yönteminiz - Ortalama)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # B) CLS Token (Standart BERT yöntemi - Temsilci Token)
        # Genelde ilk token [CLS] tokenıdır.
        cls_token = last_hidden_state[:, 0, :]

        # C) Birleştirme (Concatenation)
        # İki bilgiyi yan yana koyuyoruz: [CLS Temsili, Ortalama Temsil]
        # Boyut: (batch_size, hidden_size * 2) olur.
        combined_features = torch.cat((cls_token, mean_pooled), dim=1)

        # Sınıflandırma
        logits = self.classifier(combined_features)

        return logits