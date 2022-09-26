import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pickle import load
from textblob import TextBlob
import nltk
from PIL import Image
ps = PorterStemmer()
nltk.download('stopwords')

st.markdown("<h1 style='text-align: center; color: black;'>Sentiment anlaysis of reviews on</h1>", unsafe_allow_html=True)
image = Image.open('amazon.jpg')
col1, col2, col3 = st.columns(3)
with col1:
    st.write('')
with col2:
    st.image(image,width=300)
with col3:
    st.write('')
st.markdown("<h1 style='text-align: right; color: grey; font-size: 25px';>- Project by Team No. 5</h1>", unsafe_allow_html=True)



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"couldn\'t", "could not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def cleaning(text):
    corpus = []
    text = decontracted(text)
    text = text.lower()                              #lowering the text
    text = re.sub(r'https?\S+','',text)              #Remove the hyper link
    text = re.sub('[^a-z]',' ',text)              #Remove the character other than alphabet
    text = text.split()
    text=[ps.stem(word) for word in text if word not in stopwords.words('english')]
    text=' '.join(text)
    corpus.append(text)
    return corpus


def predict(input_msg):
    vectorizer = load(open('countvectorizer.pkl','rb'))
    
    classifier = load(open('model.pkl','rb'))
    
    clean_text = cleaning(input_msg)
    
    clean_text_encoded = vectorizer.transform(clean_text)
    
    future_text = clean_text_encoded.toarray()
    
    prediction = classifier.predict(future_text)
    
    return prediction

def main():
    st.markdown("<h2 style='text-align: center; color: black;'>Please enter your review</h2>", unsafe_allow_html=True)
    input_msg = st.text_input("")
    prediction = predict(input_msg)


    if(input_msg):
        st.subheader('Prediction')
        if prediction == 2:
            st.write("<h1 style='text-align: left; font-size: 30px; color: green;'>Positive 😁</h1>", unsafe_allow_html=True) 
        elif prediction == 0:
            st.write("<h1 style='text-align: left; font-size: 30px; color: red;'>Negative 😞</h1>", unsafe_allow_html=True)
        else:
            st.write("<h1 style='text-align: left; font-size: 30px; color: orange;'>Neutral 😑</h1>", unsafe_allow_html=True)
            
        review_pol=''.join(input_msg)
        pol_score=TextBlob(review_pol)
        polarity=round(pol_score.sentiment.polarity,4)
        st.subheader('The polarity of the review is: {}'.format(polarity))


if __name__ == '__main__' :
    main()