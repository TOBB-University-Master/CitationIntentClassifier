from transformers import AutoModel
import torch.nn as nn
import torch

class TransformerClassifier(nn.Module):
    """
        Herhangi bir Transformer tabanlı (BERT, ELECTRA, RoBERTa vb.) model için
        genel bir sınıflandırıcı. Metin girdisinden atıf niyetini tahmin eder.

        Args:
            model_name (str): Hugging Face'ten yüklenecek önceden eğitilmiş modelin adı.
            num_labels (int): Tahmin edilecek hedef etiket (citation_intent) sayısı.
            dropout_rate (float): Dropout katmanı için oran.
    """
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        # AutoModel sayesinde model_name'e göre doğru mimari (BERT, RoBERTa vb.) otomatik olarak yüklenir.
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # Farklı modellerin hidden_size'ı farklı olabilir. Bunu config'den okuyoruz.
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)


    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. Modeli (BERT, RoBERTa vb.) çalıştırmak için argümanları hazırla
        model_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # 2. Eğer 'token_type_ids' (dataset.py'den) geldiyse, argümanlara ekle.
        #    Hugging Face modeli (AutoModel), bunun 'berturk' için gerekli
        #    olduğunu bilir ancak 'xlm-roberta' için görmezden gelir.
        if token_type_ids is not None:
            model_args["token_type_ids"] = token_type_ids

        # Transformer modelinden temsilleri al
        outputs = self.transformer(**model_args)

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

        # Dropout ve sınıflandırma
        dropout_output = self.dropout(mean_pooled_output)
        logits = self.classifier(dropout_output)

        return logits