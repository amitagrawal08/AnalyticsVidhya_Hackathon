from nltk import *
import pandas as pd
from textblob import TextBlob

train_tweet_data=pd.read_csv("train_2kmZucJ.csv")
train_tweet_data.columns
## Index(['id', 'label', 'tweet'], dtype='object')

for tweets in train_tweet_data['tweet']:
    blob= TextBlob(tweets)
    print(blob.sentences)
    print(blob.sentiment.polarity, blob.sentiment.subjectivity)


    # for sentence in blob.sentences:
    #     #print(sentence.sentiment.polarity)
    #     if sentence.sentiment.polarity > 0:
    #         print('positive')
    #     elif sentence.sentiment.polarity == 0:
    #         print('neutral')
    #     else:
    #         print('negative')




