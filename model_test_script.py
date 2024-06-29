# Ensure you have already loaded your model and vectorizer
from joblib import load
clf = load('OldData/Logistic_regression.pkl')
tfidf = load('OldData/tfidfvect.pkl')

# Function to preprocess and predict
def predict_hate_speech(text):
    # Preprocess the text (ensure this is consistent with your training preprocessing)
    text = clean(text)  # Assuming 'clean' is your preprocessing function
    text_vectorized = tfidf.transform([text]).toarray()
    prediction = clf.predict(text_vectorized)
    return prediction[0]

# Test with some sample texts
sample_texts = ["Your example tweet here", "Another example tweet"]
for text in sample_texts:
    prediction = predict_hate_speech(text)
    print(f"Tweet: {text}\nPrediction: {prediction}\n")