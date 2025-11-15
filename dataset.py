# dataset.py

import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

"""
    <-- ORJİNAL SORGU (citation_context ile) --->
    select distinct (cc.citation_context_clean), LOWER(cc.section) as section, cic.citation_intent
    from cec_citation_intent cic
    join cec_citation cc on cc.id=cic.citation_id
    where cic.user_id like '48ed0fcf-4e78-4913-b96b-d942646d34b4' 
      and cc.state is null 
      and cic.citation_intent not like 'other'
      and cc.citation_context_clean not like '%<CITE>%<CITE>%'
      and cc.section is not null;
        
    <-- ORJİNAL SORGU (citation_context ile) --->
    select distinct(cc.citation_context_pre), LOWER(cc.section) as section, cic.citation_intent
    from cec_citation_intent cic
    join cec_citation cc on cc.id=cic.citation_id
    where cic.user_id like '48ed0fcf-4e78-4913-b96b-d942646d34b4' 
      and cc.state is null 
      and cic.citation_intent not like 'other'
      and cc.citation_context_clean not like '%<CITE>%<CITE>%'
      and cc.section is not null;
"""
class CitationDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 csv_path="data/train.csv",
                 db_url="mysql+pymysql://root:root@localhost:3306/ULAKBIM-CABIM-UBYT-bs",
                 max_len=256,
                 mode="labeled",
                 task = None,
                 include_section_in_input=False,
                 data_frame=None):
        """
        tokenizer: Dışarıdan verilen, önceden yüklenmiş bir tokenizer nesnesi.
        csv_path: Veri setinin yolu.
        db_url: Veritabanı bağlantı bilgisi.
        max_len: Tokenizer için maksimum uzunluk.
        mode: 'labeled' veya 'unlabeled'.
        task: None (orijinal), 'binary' (Background vs Non-Background), veya 'multiclass' (sadece Non-Background sınıfları).
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.task = task
        self.include_section_in_input = include_section_in_input

        if data_frame is not None:
            print("DataFrame (RAM) üzerinden yükleniyor...")
            self.df = data_frame
        elif os.path.exists(csv_path):
            print(f"CSV bulundu, {csv_path} dosyasından yükleniyor...")
            self.df = pd.read_csv(csv_path)
        else:
            print("CSV bulunamadı, SQL sorgusu çalıştırılıyor...")
            engine = create_engine(db_url)
            # DİKKAT: Bu sorgu sadece belirli bir kullanıcıya ait etiketli verileri çeker.
            # Tüm veriyi çekmek için sorguyu düzenlemeniz gerekebilir.
            query = """
                        SELECT c.citation_context, c.section, cic.citation_intent 
                        FROM cec_citation_intent cic
                        JOIN cec_citation c ON cic.citation_id = c.id
                        WHERE cic.user_id LIKE '48ed0fcf-4e78-4913-b96b-d942646d34b4'
                        ORDER BY c.id ASC
                    """
            self.df = pd.read_sql(query, engine)
            self.df.to_csv(csv_path, index=False)
            print(f"SQL sonucu {csv_path} olarak kaydedildi.")

        # Eksik değerleri doldur
        self.df["citation_context"] = self.df["citation_context"].fillna("")
        self.df["section"] = self.df["section"].fillna("unknown")

        # LabelEncoder ile etiketleme ---
        self.section_encoder = LabelEncoder()
        self.df['section_id'] = self.section_encoder.fit_transform(self.df['section'])

        if self.mode == "labeled":
            self.df["citation_intent"] = self.df["citation_intent"].fillna("other")
            self.label_encoder = LabelEncoder()

            # TODO: Burada farklı etiketleme  stratejileri uygulanır ---
            # Eğer task parametresi None ise, tüm etiketler kullanılır (train_v1'deki gibi).
            # Eğer task 'binary' ise, etiketler 'Background' ve 'Non-Background' olarak ikiye ayrılır.
            # Eğer task 'multiclass' ise, sadece 'Background' olmayan etiketler kullanılır.
            # ------------------------------------------------------------
            print(f"Etiketleme stratejisi: {self.task}")
            # ------------------------------------------------------------
            # Görev 1: İkili Sınıflandırma (Background vs Non-Background)
            if self.task == 'binary':
                print("İkili görev modu: Etiketler 'Background' ve 'Non-Background' olarak düzenleniyor.")
                # 'Background' olmayan tüm etiketleri 'Non-Background' olarak değiştiriyoruz.
                # NOT: Gerçek sınıf adınız 'Background' değilse burayı güncellemeniz gerekir.
                binary_labels = self.df['citation_intent'].apply(
                    lambda x: 'background' if str(x).lower() == 'background' else 'non-background')
                self.df['label_id'] = self.label_encoder.fit_transform(binary_labels)

            # Görev 2: Çok Sınıflı Uzman Sınıflandırma (Sadece Non-Background verileri)
            elif self.task == 'multiclass':
                print("Çok sınıflı görev modu: Sadece 'Non-Background' verileri kullanılıyor.")
                # DataFrame'i sadece 'Background' olmayan verilerle filtreliyoruz.
                self.df = self.df[self.df['citation_intent'] != 'background'].reset_index(drop=True)
                self.df['label_id'] = self.label_encoder.fit_transform(self.df['citation_intent'])

            # Varsayılan Durum: Tüm sınıflar kullanılır (train_v1'deki gibi)
            else:
                print("Varsayılan mod: Tüm orijinal etiketler kullanılıyor.")
                self.df['label_id'] = self.label_encoder.fit_transform(self.df['citation_intent'])


        else:
            # Etiketlenmemiş mod için sahte etiketler
            self.df['label_id'] = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label_id = row['label_id']
        section_id = row['section_id']
        text_context = str(row['citation_context'])

        # Eğer parametre True ise, bölüm başlığını ve metin içeriğini ayrı ayrı alıp tokenizer'a çift girdi olarak verilir
        # [CLS] section_title [SEP] citation_context [SEP]
        if self.include_section_in_input:
            section_title = str(row['section'])
            encoding = self.tokenizer(
                text=section_title,
                text_pair=text_context,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

        # Eğer parametre False ise sadece metin içeriğini tokenizer'a girdi olarak verilir
        # [CLS] citation_context [SEP]
        else:
            encoding = self.tokenizer(
                text = text_context,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long),
            "section_id": torch.tensor(section_id, dtype=torch.long),
            "raw_text": text_context
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item

        #return {
        #    "input_ids": encoding["input_ids"].squeeze(0),
        #    "attention_mask": encoding["attention_mask"].squeeze(0),
        #    "label": torch.tensor(label_id, dtype=torch.long),
        #    "section_id": torch.tensor(section_id, dtype=torch.long),
        #    "raw_text": text_context
        #}

    def get_label_names(self):
        """train.py'nin ihtiyaç duyduğu etiket isimlerini döndürür."""
        if hasattr(self, 'label_encoder'):
            return self.label_encoder.classes_.tolist()
        return []

    def get_section_names(self):
        """train.py'nin ihtiyaç duyduğu bölüm isimlerini döndürür."""
        if hasattr(self, 'section_encoder'):
            return self.section_encoder.classes_.tolist()
        return []