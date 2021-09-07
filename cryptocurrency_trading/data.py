from posix import environ
import pandas as pd
import requests
import os
from os import environ
from dotenv import dotenv_values


def scrape_financial_api_data():

    env_variables = dotenv_values(".env")
    api_key = env_variables['FINANCIAL_MODELLING_API_KEY']
    symbol = 'BTCUSD'
    query = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'

    response = requests.get(query)

    df = pd.DataFrame(response.json()['historical'])

    df['date'] = pd.to_datetime(df['date'])

    df_2 = df[['date', 'adjClose']]

    date = pd.to_datetime('2018-09-01')

    df3years = df_2[df_2['date'] > date]

    return df3years
