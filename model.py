from transformers import AutoModel, AutoConfig
from transformers import AutoModelForQuestionAnswering, AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForImageTextToText, AutoModelForPreTraining
import torch.nn as nn
import torch

class BerturkClassifier(nn.Module):
    """
        BERT tabanlı bir sınıflandırıcı. Metin girdisini (citation_context) ve
        kategorik bir girdiyi (section) birleştirerek atıf niyetini (citation_intent)
        tahmin eder.

        Args:
            model_name (str): Hugging Face'ten yüklenecek önceden eğitilmiş modelin adı.
            num_labels (int): Tahmin edilecek hedef etiket (citation_intent) sayısı.
            num_sections (int): Veri setindeki benzersiz bölüm (section) sayısı.
            section_embed_dim (int): Bölüm ID'leri için oluşturulacak gömülü vektörün boyutu.
            dropout_rate (float): Dropout katmanı için oran.
        """
    def __init__(self, model_name, num_labels, num_sections, section_embed_dim=50):
        super(BerturkClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # section_id'leri sayısal vektörlere dönüştürmek için Embedding katmanı
        # num_sections: Embedding tablosundaki satır sayısı (benzersiz bölüm sayısı)
        # section_embed_dim: Her bir bölüm ID'si için oluşturulacak vektörün boyutu
        self.section_embedding = nn.Embedding(num_sections, section_embed_dim)

        # Sınıflandırıcı katmanı artık BERT çıktısı (hidden_size) ile
        # section embedding çıktısının (section_embed_dim) birleşimini alacak.
        self.classifier = nn.Linear(self.bert.config.hidden_size + section_embed_dim, num_labels)


    def forward(self, input_ids, attention_mask, section_ids):
        # BERT modelinden cümle temsilini al
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # TODO: aşağıdaki satır kullanılmıyor
        pooled_output = outputs.pooler_output # [CLS] token'ının çıktısı (batch_size, hidden_size)

        # last_hidden_state -> (batch_size, 128, 768) boyutunda, PAD dahil tüm vektörler
        last_hidden_state = outputs.last_hidden_state
        # Attention mask'ı kullanarak padding token'larını hesaba katma
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Vektörleri maskeyle çarp. PAD'lerin olduğu yerler sıfırlanır.
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        # Bölücüyü bul: Gerçek token sayısı (maskenin toplamı)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # (Gerçek token'ların vektör toplamı) / (Gerçek token sayısı)
        mean_pooled_output = sum_embeddings / sum_mask
        pooled_output = mean_pooled_output

        # section_ids'leri gömülü vektörlere dönüştür
        section_embedded = self.section_embedding(section_ids) # (batch_size, section_embed_dim)

        # BERT çıktısı ile section embedding'i birleştir (yan yana ekle)
        combined_output = torch.cat((pooled_output, section_embedded), dim=1) # (batch_size, hidden_size + section_embed_dim)

        # Dropout uygula
        combined_output = self.dropout(combined_output)

        # Birleştirilmiş çıktıyı sınıflandırıcıya vererek intent tahminini yap
        intent_logits = self.classifier(combined_output)

        # Sadece intent tahminini döndür
        return intent_logits