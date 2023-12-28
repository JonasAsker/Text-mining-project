import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from transformers import pipeline, AutoTokenizer
from sklearn.metrics import accuracy_score

training_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')

multi_nb = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)

log_reg = make_pipeline(
    CountVectorizer(),
    TruncatedSVD(n_components=30, n_iter=100),
    LogisticRegression(max_iter=10000)
)

svm_model = make_pipeline(
    CountVectorizer(),
    TruncatedSVD(n_components=30, n_iter=100),
    svm.SVC()
)

print("-----------NAIVE BAYES-----------")
multi_nb.fit(training_data["lyrics"], training_data["label"])
preds = multi_nb.predict(test_data["lyrics"])
print(classification_report(test_data["label"], preds))

print("-----------LOGISTIC REGRESSION-----------")
log_reg.fit(training_data["lyrics"], training_data["label"])
preds = log_reg.predict(test_data["lyrics"])
print(classification_report(test_data["label"], preds))

print("-----------SVM-----------")
svm_model.fit(training_data["lyrics"], training_data["label"])
preds = svm_model.predict(test_data["lyrics"])
print(classification_report(test_data["label"], preds))

print("-----------BERT-----------")
tokenizer = AutoTokenizer.from_pretrained('model', model_max_length=512)
classifier = pipeline("sentiment-analysis", model="model", tokenizer=tokenizer)
preds = []

for i, row in test_data.iterrows():
    preds.append(classifier(row['lyrics'], truncation=True)[0]['label'].lower())

print(classification_report(test_data['label'], preds))