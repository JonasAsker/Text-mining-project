import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from test_train_split import split_train_test, get_even_distribution

def unlog(num):
    return 10 ** num

df = pd.read_csv('Data/songs_with_lyrics.csv')
df = df[['playlist_genre',
         'danceability',
         'energy',
         'key',
         'loudness',
         'mode',
         'tempo',
         'speechiness',
         'acousticness',
         'instrumentalness',
         'valence']]

enc = OrdinalEncoder()
enc.fit(df[['playlist_genre']])
df[['playlist_genre']] = enc.transform(df[['playlist_genre']])
df['loudness'] = df['loudness'].apply(unlog)
normalized_df=(df-df.mean())/df.std()

out = pd.cut(df['valence'], 3, labels=['negative', 'neutral', 'positive'])
df['label'] = out
df = get_even_distribution(df)
train, test = split_train_test(df)

features = ['playlist_genre','danceability','energy','key',
         'mode','tempo','speechiness',
         'instrumentalness', 'acousticness', 'loudness']

model = LogisticRegression(max_iter=10000)
model.fit(train[features], train["label"])

preds = model.predict(test[features])
print('-----------Logistic Regression-----------')
print(classification_report(test["label"], preds))

model = svm.SVC()
model.fit(train[features], train["label"])

preds = model.predict(test[features])
print('-----------SVM-----------')
print(classification_report(test["label"], preds))

model = MultinomialNB()
model.fit(train[features], train["label"])

preds = model.predict(test[features])
print('-----------Naive Bayes-----------')
print(classification_report(test["label"], preds))

model = DecisionTreeClassifier()
model.fit(train[features], train["label"])

preds = model.predict(test[features])
print('-----------Decision Tree-----------')
print(classification_report(test["label"], preds))

model = RandomForestClassifier()
model.fit(train[features], train["label"])

preds = model.predict(test[features])
print('-----------Random Forest-----------')
print(classification_report(test["label"], preds))