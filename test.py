import re
import string
import joblib,os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Ensure you have already loaded your model and vectorizer

# Vectorizer
news_vectorizer = open("NewData/tfidfvect.pkl","rb")
tfidf= joblib.load(news_vectorizer)

# Model
model = open("NewData/Logistic_regression.pkl","rb")
clf= joblib.load(model)
 # loading your vectorizer from the pkl file
 
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text




# Function to preprocess and predict
def predict_climate_sentiment(text):
    # Preprocess the text (ensure this is consistent with your training preprocessing)
    text = clean(text)  # Assuming 'clean' is your preprocessing function
    text_vectorized = tfidf.transform([text]).toarray()
    prediction = clf.predict(text_vectorized)
    return prediction[0]

# Test with some sample texts
sample_texts = ["Climate Change is a Farce! a conspiracy!, CIA cooked this up, who believes in such nonsense"]
for text in sample_texts:
    prediction = predict_climate_sentiment(text)
    print(f"Tweet: {text}\nPrediction: {prediction}\n")
