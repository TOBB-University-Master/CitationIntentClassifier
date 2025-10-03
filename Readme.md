# Citation Intent Classifier

Bu proje, Türkçe akademik metinlerdeki atıf cümlelerini analiz ederek atıf niyetini (citation intent) belirlemeye 
yöneliktir. 

Kullanılan modeller aşağıdaki gibi listelenmektedir.
* dbmdz/bert-base-turkish-cased
* dbmdz/electra-base-turkish-cased-discriminator
* xlm-roberta-base
* microsoft/deberta-v3-base


## 🎓 Amaç

Türkçe atıf cümlelerinin şu 6 sınıftan birine ait olup olmadığını sınıflandırmak:

* background
* discuss
* basis
* support
* differ
* other

## 🔧 Kurulum

Python 3.9+ ve PyTorch ile uyumlu bir conda ortamında:

```bash
  conda create --name <ortam_adi> python=3.9
  conda activate <ortam_adi>
  pip install -r requirements.txt
```

## 📂 Klasör Yapısı

```
base_directory/
├── train_v1.py                    # Standart sınıflandırma için eğitim (model değişken olarak verilebilir)
├── train_v2.py                    # Hiyerarşik sınıflandırma için eğitim (model değişken olarak verilebilir)
├── checkpoints/                   # Kaydedilen model ve optimizer durumları
├── predict_v1.py                  # Standart sınıflandırma için tahmin (model değişken olarak verilebilir)
├── predict_v1.py                  # Hiyerarşik sınıflandırma için tahmin (model değişken olarak verilebilir)
├── generic_model.py               # Generic TransformerClassifier sınıfıdır. Parametre olarak model ismi alır ve sonunda sınıflandırıcısı vardır
├── dataset.py                     # SQL veya CSV tabanlı veri çekme
├── model.py                       # [DEPRECIATE] BERTurkClassifier modeli
├── train.py                       # [DEPRECIATE] Eğitme döngüsü, checkpoint desteği ile
├── extract_embeddings.py          # CLS, mean, max, attention-weighted embedding çıkışı
├── analyze_clusters.py            # Elbow, silhouette ve t-SNE analizleri
├── cls_to_closest_tokens.py       # CLS embedding ile en yakın tokenları bulma
├── embedding_to_closest_tokens.py # Tüm embedding'ler için yakın token analizi
├── predict_untrained.py           # Eğitim öncesi embedding'leri gözlemleme
├── output/                        # .npy dosyaları ve analiz grafikleri
├── data/
│   └── train.csv                  # SQL'den çekilen veya dışa aktarılan veri
│   └── data_v1.csv                # SQL'den çekilen veya dışa aktarılan temizlenmiş veri
```

## 📃 Veri Kaynağı

Veriler MySQL veritabanından şu sorgularla alınır:

* Etiketli: `cec_citation_intent` üzerinden
* Etiketsiz: `cec_citation` tablosundan

> CSV varsa tekrar SQL sorgusu çalışmaz; `CitationDataset` otomatik kontrol eder.

## 📅 Eğitim Aşamaları

***train_v1.py:*** Bir metnin atıf niyetini tüm sınıflar arasında doğrudan tek bir adımda sınıflandıran standart bir 
Transformer modelini eğitir.

***train_v2.py:*** Atıf niyetini iki aşamalı hiyerarşik bir yaklaşımla sınıflandıran iki ayrı model eğitir; ilk model 
metnin "background" olup olmadığını anlarken, ikinci model "non-background" metinleri kendi alt türlerine ayırır.

```bash
  python train_v1.py
  python train_v2.py
```

Model `checkpoints/berturk_classifier_checkpoint.pt` dosyasına kaydedilir ve tekrar çalıştırıldığında kaldığı yerden devam eder.

## 📈 Embedding Türleri

```bash
python extract_embeddings.py
```

Aşağıdaki türler çıkarılır ve `output/` klasörüne zaman etiketli olarak kaydedilir:

* CLS (pooler\_output)
* Mean pooling
* Max pooling
* Attention-weighted pooling

Her bir `.npy` dosyası `202505201434_cls_embedding.npy` gibi timestamp'li şekilde adlandırılır.

## 🔍 Analiz

```bash
python analyze_clusters.py
```

* Her embedding türü için t-SNE, silhouette score ve elbow grafiği oluşturur
* `output/` klasörüne `tsne`, `silhouette`, `elbow` görselleri kaydedilir

## 🔁 Alternatif Analizler

```bash
python TEST_cls_to_closest_tokens.py
python embedding_to_closest_tokens.py
```

Verilen cümle embedding'lerinin sözlükteki en yakın token'larla benzerlik analizi yapılır.

## 🚩 Notlar

* Model BERTurk: `dbmdz/bert-base-turkish-cased`
* Atıf intent sınıflandırması 6 sınıflı softmax ile yapılır
* Checkpoint: model, optimizer, scheduler ve epoch bilgilerini içerir
* `CitationDataset`: CSV varsa onu kullanır, yoksa SQL sorgusu çalıştırır ve CSV oluşturur

---

Bu proje, Türkçe akademik metinlerde yapay zekâ tabanlı anlam çıkarımı çalışmasının bir parçasıdır.
