# scrapping daily bitcoin prices from financial modelling prep

import requests
import pandas as pd

api_key='3c63db1acd4399020c7f6eee92ec77cf'
symbol='BTCUSD'
query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'

response=requests.get(query)

btcdaily=pd.DataFrame(response.json()['historical'])
btcdaily.to_csv('btsdaily', encoding='utf-8', index=False)
pd.read_csv('btcdaily')
