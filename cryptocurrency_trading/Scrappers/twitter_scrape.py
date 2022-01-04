#imports
from IPython.core.display import display, HTML
display(HTML('<style>.container { width:70% !important; }</style>'))

import os
import requests
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
import time
from dotenv import load_dotenv


#scraping tweets

def scrape_tweets(start, end):
    load_dotenv('.env')
    token=os.getenv('tw_api')
    dt_range = pd.date_range(start = start, end = end)
    dt_range = pd.DataFrame(dt_range, columns=['date'])


    url = 'https://api.twitter.com/2/tweets/search/all'
    header = {"Authorization": f"Bearer {token}"}



    start = dt_range['date'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt_range['date'].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")

    results = []
    for idx in tqdm(dt_range.index):
        start = dt_range['date'].iloc[idx].strftime("%Y-%m-%dT%H:%M:%SZ")
        if idx == dt_range.index[-1]:
            end = (datetime.utcnow()-pd.Timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end = dt_range['date'].iloc[idx+1].strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
        "query":'bitcoin',
        "start_time":f"{start}",
        "end_time":f"{end}",
        "max_results":5
        }

        error_counter = 0
        while True:
            try:
                response = requests.get(url, headers = header, params = params)
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

    df= pd.concat(results, axis = 0).reset_index(drop = True)

    return df

#scrapping tweets_count

def get_tweets_count(start, end):
    load_dotenv('.env')
    token=os.getenv('tw_api')
    dt_range = pd.date_range(start = start, end = end)
    dt_range = pd.DataFrame(dt_range, columns=['date'])

    start = dt_range['date'].iloc[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt_range['date'].iloc[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
    print(start)
    print(end)

    url = 'https://api.twitter.com/2/tweets/counts/all'
    header = {"Authorization": f"Bearer {token}"}

    results = []
    for i in tqdm(range(0, len(dt_range), 30)):
        start = dt_range['date'].iloc[i].strftime("%Y-%m-%dT%H:%M:%SZ")
        if i+29 < len(dt_range):
            end = dt_range['date'].iloc[i+29].strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end = (datetime.utcnow()-pd.Timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "query":'bitcoin',
            "start_time":f"{start}",
            "end_time":f"{end}",
            "granularity":'day'
            }
        response = requests.get(url, headers = header, params = params)
        results.append(pd.DataFrame(response.json()['data']))
        time.sleep(.5)

    df_count = pd.concat(results, axis = 0).reset_index(drop = True)
    df_count['tweet_count'] = df_count['tweet_count'].astype(int)

    return df_count
