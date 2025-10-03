from transformers import AutoModel
import torch.nn as nn
import torch

class BerturkClassifier(nn.Module):
    """
        BERT tabanlı bir sınıflandırıcı. Metin girdisini (citation_context) ve
        kategorik bir girdinin atıf niyetini (citation_intent) tahmin eder.

        Args:
            model_name (str): Hugging Face'ten yüklenecek önceden eğitilmiş modelin adı.
            num_labels (int): Tahmin edilecek hedef etiket (citation_intent) sayısı.
            dropout_rate (float): Dropout katmanı için oran.
    """
    def __init__(self, model_name, num_labels):
        super(BerturkClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # Sınıflandırıcı katmanı artık sadece BERT çıktısını (hidden_size) girdi olarak alacak.
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)


    def forward(self, input_ids, attention_mask):
        # BERT modelinden cümle temsilini al
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

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

        # Dropout uygula
        dropout_output = self.dropout(pooled_output)

        # Özetlenmiş çıktıyı sınıflandırıcıya vererek intent tahminini yap
        intent_logits = self.classifier(dropout_output)

        return intent_logits