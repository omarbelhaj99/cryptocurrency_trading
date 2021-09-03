#imports
import pandas as pd
from datetime import datetime
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request



def resampling(df,start_date):
    df['start']=pd.to_datetime(df['start']).dt.date
    df['start']=pd.to_datetime(df['start'])

    day=df['start'].nunique()

    # start_date = datetime(2018,1,1)

    e=[]

    i=0
    for j in pd.date_range(start_date, periods=day):
        print(j,i)
        if df[df['start']==j]['id'].count()>=100:
            e.append(df[df['start']==j].sample(n=100))

    resampled_df=pd.concat(e)
    resampled_df=resampled_df.reset_index().drop('index',axis=1)


    return resampled_df





def get_tweets_sentimentscore(resampled_df):

    def preprocess(text):
        new_text = []


        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)


    task='sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    labels=[]
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)


    def sentimentscore(text):

            # text = "Good night 😊"
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)



        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            # print(f"{i+1}) {l} {np.round(float(s), 4)}")

        return (scores[2]-scores[0])/scores[1]

    resampled_df['senntimentscore']=''

    for i in range(10):
        resampled_df['sentimentscore'][i]=sentimentscore(resampled_df.text[i])
        print(i)

    return resampled_df


def get_daily_sentimentscore(df):
    return df[['start','sentimentscore']].groupby(['start']).mean()
