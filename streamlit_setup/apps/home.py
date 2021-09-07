from pathlib import PureWindowsPath
import streamlit as st

import numpy as np
import pandas as pd
from PIL import Image
import datetime
#!/usr/bin/env python
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import json

### THIS WILL GO ON THE FIRST PAGe


def get_jsonparsed_data():
    """
    Receive the content of ``url``, parse it as JSON and return the object.
    Parameters
    ----------
    url : str
    Returns
    -------
    dict
    """
    url = ("https://financialmodelingprep.com/api/v3/quote/BTCUSD?apikey=a58413697e8263de9c95cab92049ea3f")
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def app():

    st.markdown("""# Cryptocurrency trading!""")
    st.write("""## We predict the movement in bitcoin price, taking into account sentiment analysis from reddit and twitter""")
    st.write("""Do you wanna have a go yourself?""")

    st.write('This page tells you some stuff about our model')


    # print(get_jsonparsed_data(url)[0]['price'])
    bitcoin_live_data = get_jsonparsed_data()[0]
    bitcoin_current_price = bitcoin_live_data ['price']
    bitcoin_change = bitcoin_live_data ['changesPercentage']


    # number = st.number_input('How much money are you investing???')

    # st.write('You are investing: ', number)


    col1, col2, col3 = st.columns(3)
    col1.metric("", "", "")
    col2.metric("BITCOIN", f"${bitcoin_current_price}", f"{bitcoin_change}%")
    col3.metric("", "", "")
    # perhaps insert here the current value of bitcoin?

