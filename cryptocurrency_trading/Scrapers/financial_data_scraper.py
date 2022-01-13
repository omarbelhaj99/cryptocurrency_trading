# scrapping daily bitcoin prices from financial modelling prep

import requests
import pandas as pd

def get_price_data(startdate):
    # env_variables = dotenv_values(".env")
    # print(env_variables)
    api_key = 'a58413697e8263de9c95cab92049ea3f'
    # api_key = env_variables['FINANCIAL_MODELLING_API_KEY']
    symbol='BTCUSD'
    query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'
    response=requests.get(query)
    df = pd.DataFrame(response.json()['historical'])
    df['date'] = pd.to_datetime(df['date'])
    df_2 = df[['date', 'adjClose','volume']]
    startdate = pd.to_datetime(startdate)
    dfpricedata = df_2[df_2['date'] >= startdate]
    return dfpricedata
