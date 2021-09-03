import requests
import pandas as pd

api_key=''
symbol='BTCUSD'
query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'

response=requests.get(query)

btsdaily = pd.DataFrame(response.json()['historical'])

btsdaily['date'] = pd.to_datetime(btsdaily['date'])

btsdaily.to_csv('btsdaily', encoding='utf-8', index=False)

pd.read_csv('btsdaily')
