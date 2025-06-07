import joblib
import os

# Model ve vectorizer yolu
MODEL_PATH = 'models/svm_model.pkl'
VECTORIZER_PATH = 'models/vectorizer.pkl'

# Model ve vectorizer'i yükle
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment(title, body):
    """
    Yorum başlığı ve içeriğine göre olumlu/olumsuz tahmini döndürür.
    """
    text = f"{title} {body}".strip()
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Olumlu" if prediction == 1 else "Olumsuz"

# Örnek kullanım
if __name__ == "__main__":
    title = input("Yorum başlığı: ")
    body = input("Yorum içeriği: ")
    result = predict_sentiment(title, body)
    print(f"Tahmin: {result}")
