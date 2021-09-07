#imports
from re import S
import pandas as pd
from datetime import datetime
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
#
from IPython.core.display import display, HTML
display(HTML('<style>.container { width:70% !important; }</style>'))

import os
import requests
from tqdm.notebook import tqdm
import time
from dotenv import load_dotenv




##
def scrape_tweets(start, end):
    load_dotenv('.env')
    token=os.getenv('tw_api')
    dt_range = pd.date_range(start=start, end=end)
    dt_range = pd.DataFrame(dt_range, columns=['date'])

    url = 'https://api.twitter.com/2/tweets/search/all'
    header = {"Authorization": f"Bearer {token}"}

    start = dt_range['date'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt_range['date'].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")

    results = []
    for idx in tqdm(dt_range.index):
        start = dt_range['date'].iloc[idx].strftime("%Y-%m-%dT%H:%M:%SZ")
        if idx == dt_range.index[-1]:
            end = (datetime.utcnow() -
                   pd.Timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end = dt_range['date'].iloc[idx + 1].strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "query": 'bitcoin',
            "start_time": f"{start}",
            "end_time": f"{end}",
            "max_results": 10
        }

        error_counter = 0
        while True:
            try:
                response = requests.get(url, headers=header, params=params)
                df_temp = pd.DataFrame(response.json()['data'])
                df_temp['start'] = start
                df_temp['end'] = end
                results.append(df_temp)
                break
            except:
                error_counter += 1
                if error_counter == 5:
                    print(f'skipping - {start}')
                    break
                time.sleep(2)

        time.sleep(.5)

    df = pd.concat(results, axis=0).reset_index(drop=True)

    return df


#scrapping tweets_count


def get_tweets_count(start, end):
    load_dotenv('.env')
    token=os.getenv('tw_api')
    dt_range = pd.date_range(start=start, end=end)
    dt_range = pd.DataFrame(dt_range, columns=['date'])

    start = dt_range['date'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt_range['date'].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    print(start)
    print(end)

    url = 'https://api.twitter.com/2/tweets/counts/all'
    header = {"Authorization": f"Bearer {token}"}

    results = []
    counter = 0
    for i in tqdm(range(0, len(dt_range), 30)):
        start = dt_range['date'].iloc[i].strftime("%Y-%m-%dT%H:%M:%SZ")
        if i + 29 < len(dt_range):
            end = dt_range['date'].iloc[i + 29].strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end = (datetime.utcnow() -
                   pd.Timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "query": 'bitcoin',
            "start_time": f"{start}",
            "end_time": f"{end}",
            "granularity": 'day'
        }
        response = requests.get(url, headers=header, params=params)
        results.append(pd.DataFrame(response.json()['data']))
        time.sleep(.5)

    df_count = pd.concat(results, axis=0).reset_index(drop=True)
    df_count['tweet_count'] = df_count['tweet_count'].astype(int)

    return df_count


##
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

def download_model():
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
    return 'model downloaded'



def get_tweets_sentimentscore(df):
    df['sentimentscore'] = ''

    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    labels = []
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)


    def preprocess(text):
        new_text = []


        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

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



    for i in range(len(df)):
        df['sentimentscore'][i]=sentimentscore(df.text[i])
        print(i)

    df = df[['start', 'sentimentscore']]

    df['start'] = pd.to_datetime(df['start']).dt.strftime('%Y-%m-%d')
    df['sentimentscore'] = pd.to_numeric(df['sentimentscore'],errors='ignore')
    df= df.groupby('start').mean()

    return df


def get_sentiment_and_count(start, end):
    df= scrape_tweets(start,end)
    df= get_tweets_sentimentscore(df)

    df1 = get_tweets_count(start, end)
    df1['start'] = pd.to_datetime(df1['start']).dt.strftime('%Y-%m-%d')

    results = pd.merge(df, df1, how='left', on='start')
    results=results.drop('end',axis=1)

    return results
