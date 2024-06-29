import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import re
import string
from joblib import dump

# Building classification models
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv("train.csv")

# Preprocessing
data["sentiment"] = data["sentiment"].map({0: "Neutral", 1: "Pro", 2: "News", -1: "Anti"})
data = data[["message", "sentiment"]]

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

print('.. cleaning data')
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
data["message"] = data["message"].apply(clean)
print('... done data cleaming')

# Vectorization
x = np.array(data["message"])
y = np.array(data["sentiment"])
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(x)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Model Training
print('..training model')
clf = LinearSVC()
clf.fit(X_train, y_train)
print('done training model')

# Saving the model and vectorizer
print('... creating model pickle file')
dump(clf, 'LinearSVC.pkl')
dump(tfidf, 'tfidfvect.pkl')
print("...done dumping pickle file")
print('done model building')