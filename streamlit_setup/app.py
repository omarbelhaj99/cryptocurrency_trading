import streamlit as st

import numpy as np
import pandas as pd
from PIL import Image
import datetime



st.set_page_config(
            page_title="Quick reference", # => Quick reference - Streamlit
            page_icon="🐍",
            layout="centered", # wide
            initial_sidebar_state="auto") # collapsed

df = pd.DataFrame({
    'first column': ['Some general info please' , 'Check out past performance', 'Give me tradding advice'],
    'second column': [10, 20, 30]
})
option = st.sidebar.selectbox(
    'What do you want to use our services for?',
     df['first column'])

'You selected:', option


st.markdown("""# Cryptocurrency trading!
## We predict the movement in bitcoin price, taking into account sentiment analysis from reddit and twitter
Do you wanna have a go yourself?""")

number = st.number_input('How much money are you investing???')

st.write('You are investing: ', number)

col1, col2, col3 = st.columns(3)
col1.metric("SPDR S&P 500", "$437.8", "-$1.25")
col2.metric("FTEC", "$121.10", "0.46%")
col3.metric("BTC", "$46,583.91", "+4.87%")
# perhaps insert here the current value of bitcoin?

st.markdown("""## Past performance """)

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
min_start = datetime.datetime(2021, 6, 1)
start_date = st.date_input('Start date', min_start, min_value = min_start, max_value = yesterday)
end_date = st.date_input('End date', today, min_value  = min_start+datetime.timedelta(days=1), max_value = today)
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')


# from model
buy = 0

st.markdown("""## Using the model """)
if st.button('Show me what to do with my money!'):
    # print is visible in the server output, not in the page
    print('The model will tell you what to do')
    # st.write('I was clicked 🎉')
    # st.write('Further clicks are not visible but are executed')

    if buy == 1:
        image = Image.open('photos_frontend/Buy-Bitcoin.jpg')
    else:
        image = Image.open('photos_frontend/sell-bitcoin.jpg')
    st.image(image, caption='Our recommendation!!')
    st.write('👎SELL!!👎')
    st.write('👍BUY!!👍')




else:
    st.write('Will it be sell or buy!!')


CSS = """
h1 {
    color: red;
}

"""

if st.checkbox('Inject CSS'):
    st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

df = pd.DataFrame({
          'first column': list(range(1, 11)),
          'second column': np.arange(10, 101, 10)
        })

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('Select a line count', 1, 10, 3)

# and used in order to select the displayed lines
head_df = df.head(line_count)

head_df

