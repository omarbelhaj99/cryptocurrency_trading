#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd


# In[2]:


api_key='a58413697e8263de9c95cab92049ea3f'
symbol='BTCUSD'
query=f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key}'


# In[3]:


response=requests.get(query)


# In[4]:


pd.DataFrame(response.json()['historical'])


# In[ ]:




