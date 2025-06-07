# SVM Amazon Yorum Sınıflandırma Projesi

Bu proje, Amazon'daki ürün yorumlarını kullanarak makine öğrenmesiyle yorumların **olumlu** veya **olumsuz** olup olmadığını tahmin eder. Model olarak **SVM (Support Vector Machine)** kullanılmıştır.

## 📊 Proje Yapısı

```
svm-amazon-review-classifier/
│
├── data/
│   └── 20191226-reviews.csv         # Ham veri seti
│
├── models/
│   ├── svm_model.pkl                # Eğitilmiş model
│   └── vectorizer.pkl               # TF-IDF dönüştürücü
│
├── src/
│   ├── train.py                     # Model eğitimi
│   └── predict.py                   # Tahmin scripti
│
├── requirements.txt                   # Bağımlılıklar
├── README.md                          # Bu dosya
└── .gitignore                         # Git dışı dosyalar
```

## ⚙️ Kurulum

1. Bu repoyu klonla:

```bash
git clone https://github.com/kullaniciadi/svm-amazon-review-classifier.git
cd svm-amazon-review-classifier
```

2. Gerekli paketleri kur:

```bash
pip install -r requirements.txt
```

## 🧑‍💻 Modeli Eğitme

```bash
python src/train.py
```

Eğitilen model ve TF-IDF dönüştürücü `models/` klasörüne kaydedilir.

## ✅ Tahmin Yapma

```bash
python src/predict.py
```

Terminalde yorum başlığı ve içeriği girerek tahmin sonucunu alabilirsiniz:

```
Yorum başlığı: Harika bir telefon
Yorum içeriği: Çekim gücü mückemmel, tavsiye ederim.
Tahmin: Olumlu
```

## 📈 Performans

- Doğruluk: %93.2
- Model: LinearSVC
- Vektörleştirme: TF-IDF

## ✉️ Lisans

Bu proje MIT lisansı altındadır.
