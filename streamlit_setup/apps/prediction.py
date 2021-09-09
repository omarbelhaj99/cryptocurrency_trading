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

    import json
from cryptocurrency_trading import trainer


def obtain_prediction(start_date, end_date):
    st.write('predicting')
    y_pred = trainer.predict_price(start_date,end_date)
    buy = y_pred>0
    return buy


def app():
    # end = datetime.date.today()
    # start = end - datetime.timedelta(1)
    start=datetime.datetime(2021,9,1)
    end=datetime.datetime(2021,9,7)
    #buy = obtain_prediction(start, end)

    st.markdown("""# Time to get rich! """)
    st.write("### You've invested in crypto?? We tell you what to do with your money!")
    st.write("I would trust us, our model is more accurate than you!!")

    if st.button('Show me what to do with my money!'):
        # print is visible in the server output, not in the page
        print('The model will tell you what to do')
        # st.write('I was clicked 🎉')
        # st.write('Further clicks are not visible but are executed')
        buy = obtain_prediction(start, end)
        if buy == 1:
            say = 'buy'
            image = st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1631179116/699-cryptocurrency/Buy-Bitcoin_aciwoa.jpg', caption='Our recommendation!!')
        else:
            say = 'sell'
            image = st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1631179116/699-cryptocurrency/sell-bitcoin_ysvl9r.jpg', caption='Our recommendation!!')
        #st.image(image, caption='Our recommendation!!')
        st.write(f"RUN!!! go {say} bitcoin!!!! Tomorrow you will have more money💰💰💰")
        st.write('')
        st.write('If you need a little more information to trust us go ahead to our Past Performance window and check our our previous success 😉')
        # st.write('👎SELL!!👎')
        # st.write('👍BUY!!👍')

    else:
        st.write('Will it be sell or buy!! Go on find out!')
