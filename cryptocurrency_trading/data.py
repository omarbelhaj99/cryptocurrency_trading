from posix import environ
import pandas as pd
import requests
import os
from os import environ
from dotenv import dotenv_values

from red_sentiment import all_reddit
from technical_analysis import all_tech_analysis
from tw_sentiment import get_sentiment_and_count

def get_price_data():
    env_variables = dotenv_values(".env")
    api_key = env_variables['FINANCIAL_MODELLING_API_KEY']
    symbol='BTCUSD'
    query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'
    response=requests.get(query)
    df = pd.DataFrame(response.json()['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df_2 = df[['date', 'adjClose']]
    date = pd.to_datetime('2018-09-01')
    df3years = df_2[df_2['date']>date]
    return df3years


def get_all_data():

    df = get_price_data()
    reddit_data = all_reddit(start_date, end_date)
    tech_data = all_tech_analysis(df)

    start=start_date
    end=end_date
    twitter_data=get_sentiment_and_count(start,end,token)