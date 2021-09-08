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

def app():
    buy = 1

    st.markdown("""# Time to get rich! """)
    st.write("### You've invested in crypto?? We tell you what to do with your money!")
    st.write("I would trust us out model is 90 percent accurate!")

    if st.button('Show me what to do with my money!'):
        # print is visible in the server output, not in the page
        print('The model will tell you what to do')
        # st.write('I was clicked 🎉')
        # st.write('Further clicks are not visible but are executed')

        if buy == 1:
            say = 'buy'
            image = Image.open('photos_frontend/Buy-Bitcoin.jpg')
        else:
            say = 'sell'
            image = Image.open('sell-bitcoin.jpg')
        st.image(image, caption='Our recommendation!!')
        st.write(f"RUN!!! go {say} bitcoin!!!! Tomorrow you will have more money💰💰💰")
        st.write('')
        st.write('If you need a little more information to trust us go ahead to our Past Performance window and check our our previous success 😉')
        # st.write('👎SELL!!👎')
        # st.write('👍BUY!!👍')




    else:
        st.write('Will it be sell or buy!! Go on find out!')
