from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt

def get_all_data():
    with open("train_2kmZucJ.csv", encoding="utf8") as csv_file:
        data = csv_file.read().split('\n')
        return data

def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 3 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))
    return processing_data

all_data=get_all_data()
processing_data=preprocessing_data(all_data)

def training_step(data, vectorizer):
    training_text = [data[2] for data in data]
    training_result = [data[1] for data in data]
    training_text = vectorizer.fit_transform(training_text)
    return BernoulliNB().fit(training_text, training_result)

vectorizer = CountVectorizer(binary = 'true')
classifier = training_step(processing_data, vectorizer)
result = classifier.predict(vectorizer.transform(["Happy"]))
result[0]

def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    print(text, ":", print_text)

for i in processing_data:
    print_result(analyse_text(classifier, vectorizer, i[2]))
    print(i[0])





# test_tweet_data=pd.read_csv("test_oJQbWVk.csv")
# test_tweet_data.columns
# data1=test_tweet_data['tweet']
# data1
# processing_data1 = []
# for single_data1 in data1:
#     processing_data1.append(single_data1)
#     #print(single_data1)
# processing_data1
#
# for evaluation_text in processing_data1:
#     analysis_result = analyse_text(classifier, vectorizer, evaluation_text)
#     text, result = analysis_result
#     print (text, result)

