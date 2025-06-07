import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Dosya yolları
DATA_PATH = 'data/20191226-reviews.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Klasör yoksa oluştur
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Veriyi yükle
df = pd.read_csv(DATA_PATH)

# 2. Sadece 1-2 ve 4-5 puanları al, 3 puanlıları dışarda bırak
df = df[df['rating'].isin([1, 2, 4, 5])].copy()
df['label'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# 3. Metin hazırla
df['text'] = (df['title'].fillna('') + ' ' + df['body'].fillna('')).str.strip()

# 4. Veri ayır
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Modeli eğit
model = LinearSVC()
model.fit(X_train_vec, y_train)

# 7. Modeli ve vectorizer'i kaydet
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model ve TF-IDF vectorizer 'models/' klasörüne kaydedildi.")
