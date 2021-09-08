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

    st.markdown("""## Using the model """)

    if st.button('Show me what to do with my money!'):
        # print is visible in the server output, not in the page
        print('The model will tell you what to do')
        # st.write('I was clicked 🎉')
        # st.write('Further clicks are not visible but are executed')

        if buy == 1:
            image = Image.open('photos_frontend/Buy-Bitcoin.jpg')
        else:
            image = Image.open('sell-bitcoin.jpg')
        st.image(image, caption='Our recommendation!!')
        st.write('👎SELL!!👎')
        st.write('👍BUY!!👍')




    else:
        st.write('Will it be sell or buy!!')
