#Import to ML (scikit-learn), Data (Pandas) and Math (NumPy) Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Data Loading from 'Kaggle'
df = pd.read_csv('language_detection.csv')

df.head(5)

"""### Value Counts of Language Label"""

df["language"].value_counts()

"""### Value Counts of Text Feature"""

df["Text"].value_counts()

"""> # *Process*
> * **Import to ML (scikit-learn) Libraries**
> * **Data Preprocessing**
> * **NLP system entegration to Data**
> * **Model Creating**
"""

#Import to ML (scikit-learn) Libraries
from sklearn.naive_bayes import MultinomialNB #RidgeClassifier

#Data Preprocessing

#NLP system entegration to Data    
X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(df.Text)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)

#Model Creating
model = MultinomialNB()

model.fit(X_train_tfidf, df.language)

"""> # *Model Accuracy Score*
> * **model.score([Test_data])**
"""

print("Model Accuracy:", model.score(X_CountVectorizer.transform(X_train_counts),df.language))

"""> # *Prediction*"""

#Data of Prediction
text = input("Please, Enter your idea: ")

text = [text]

text_counts = X_CountVectorizer.transform(text)

#Prediction Processing
prediction = model.predict(text_counts)

f"Prediction is {prediction[0]}"