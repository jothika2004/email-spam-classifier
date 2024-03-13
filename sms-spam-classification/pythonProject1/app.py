import streamlit as st

import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def transform_text(Text):
    Text=Text.lower()
    Text=nltk.word_tokenize(Text)
    y=[]
    for i in Text:
        if i.isalnum():
            y.append(i)
    Text=y[:]
    y.clear()
    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    Text=y[:]
    y.clear()
    for i in Text:
        y.append(ps.stem(i))
    return " ".join(y)
with open('vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)
with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

st.title("Email/sms spam classifier")
sms=st.text_area("Enter the message:")
if st.button('Predict'):
    transformed_sms=transform_text(sms)
    vector_input=tfidf.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")

