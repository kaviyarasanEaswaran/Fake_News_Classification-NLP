import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import streamlit as st

# ================Project_Title============#
styled_text = "<p style='font-size: 30 px; font-weight: bold;color:magenta;'>Fake News Classification:</p>"
st.markdown(styled_text, unsafe_allow_html=True)

df = pd.read_csv(r"C:\Users\arjun\PycharmProjects\pythonProject\News_dataset.csv")
df.drop(["author","id","text"],axis=1,inplace = True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True,drop=True)
lm = WordNetLemmatizer()
stopwords = stopwords.words('english')
corpus = []
for i in range (len(df)):
    review = re.sub('^a-zA-Z0-9',' ', str(df['title'][i]))
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(data) for data in review if data not in stopwords]
    review = " ".join(review)
    corpus.append(review)
tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()
filename = r"C:\Users\arjun\PycharmProjects\pythonProject\Fake_News_Classification4.sav"
loaded_model = joblib.load(open(filename, 'rb'))
corpus1 = []
news = st.text_input("Enter the news", value="")
submit = st.button("Analyse the news ")
if submit:
    review = re.sub('[^a-zA-Z0-9]', ' ', news)
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(data) for data in review if data not in stopwords]
    review = " ".join(review)
    corpus1.append(review)
    x = tf.transform(corpus1).toarray()
    predict1 = loaded_model.predict(x)

    if predict1[0]== 1:
        st.success( "The News Is Fake")
        

    else:
        st.success("The News Is Real")
