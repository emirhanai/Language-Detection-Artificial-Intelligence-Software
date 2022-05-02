# -*- coding: utf-8 -*-
"""Language Detection Artificial Intelligence Software.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i9uo3oKqBmzpR_DnJRO8Lz8kb9Wvfwdg

> # Language Detection Artificial Intelligence Software - By Emirhan BULUT

I developed 'Language Detection Artificial Intelligence Software'. I share this software with all humanity for free from Kaggle and GitHub.
The sensitivity score is quite high and the accuracy score is in a very advantageous position. It has 97% accuracy and 98.4% sensitivity score.
I share Dataset and Artificial Intelligence Software as open source. I used the Naive Bayes algorithm and manipulated the dataset to improve the algorithm. In this way, the sensitivity rate and score of the software has been increased. In addition, I adapted the data according to the algorithm and made the algorithm work. In this study, it was studied on the data, not on the algorithm.

I wish conveniences,

Emirhan BULUT

Senior Artificial Intelligence Engineer

###**The coding language used:**

`Python 3.9.8`

###**Libraries Used:**

`NumPy`

`Pandas`

`Scikit-learn (SKLEARN)`

<img class="fit-picture"
     src="https://github.com/emirhanai/Language-Detection-Artificial-Intelligence-Software/blob/main/Language%20Detection%20Artificial%20Intelligence%20Software.png?raw=true"
     alt="Language Detection Artificial Intelligence Software - Emirhan BULUT">
     
### **Developer Information:**

Name-Surname: **Emirhan BULUT**

Contact (Email) : **emirhan@isap.solutions**

LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**

[LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/

Kaggle: **[https://www.kaggle.com/emirhanai][Kaggle]**

Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**

[Kaggle]: https://www.kaggle.com/emirhanai

[OfficialWebSite]: https://www.emirhanbulut.com.tr

> # Introduction to Language Detection Artificial Intelligence Software
> * **Import to ML (scikit-learn), Data (Pandas) and Math (NumPy) Libraries**
> * **Data Loading from 'Kaggle'**
"""

#Import to ML (scikit-learn), Data (Pandas) and Math (NumPy) Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(df.Text, 
                                                    df.language,
                                                    test_size=0.325000000000000001,
                                                    random_state=2551,
                                                    shuffle=True)
#NLP system entegration to Data    
X_CountVectorizer = CountVectorizer(stop_words='english')

X_train_counts = X_CountVectorizer.fit_transform(X_train)

X_TfidfTransformer = TfidfTransformer()

X_train_tfidf = X_TfidfTransformer.fit_transform(X_train_counts)

#Model Creating
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)

"""> # *Model Accuracy Score*
> * **model.score([Test_data])**
"""

model.score(X_CountVectorizer.transform(X_test),y_test)

"""> # *Prediction*"""

#Data of Prediction
text = """I quite like him. 
I'm so in love with him and my heart flutters when I see him."""

text = [text]

text_counts = X_CountVectorizer.transform(text)

#Prediction Processing
prediction = model.predict(text_counts)

f"Prediction is {prediction[0]}"