import streamlit as st
from joblib import load

# Load the model and vectorizer
clf = load('climate_change_model.pkl')
cv = load('count_vectorizer.pkl')

def hate_speech_detection(tweet):
    data = cv.transform([tweet]).toarray()
    prediction = clf.predict(data)
    return prediction[0]

st.title("Climate Change Clasification")
user_input = st.text_area("Enter a Tweet:")

if user_input:
    prediction = hate_speech_detection(user_input)
    st.write(f"Prediction: {prediction}")