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



def plot_bitcoin_change(df):
    st.markdown("""### Bitcoin Price """)
    st.write('Here you can see the change of bitcoin price against of that our model predicted!')
    st.write('(actually now is open against close but soon you will be able to)')
    st.line_chart(df)

def get_past_data(start, end):
    #obtain past csv data
    past_performance = pd.read_csv('../cryptocurrency_trading/data/4-month-BTC-perf.csv')
    past_performance['date'] = pd.to_datetime(past_performance['date'])
    past_performance.sort_values(by="date", inplace = True)
    past_performance['date'] = past_performance['date'].dt.date
    past_performance.set_index('date', inplace = True)
    df = past_performance[past_performance.index >= start]
    df = df[df.index <= end].copy()
    return df

def get_earnings(investment, earnings,start):
    yesterday  = start - datetime.timedelta(days=1)
    initial_investment = pd.DataFrame([investment], index = [yesterday], columns = ['dif'])
    earnings.columns = ['pred', 'real']
    earnings['dif'] = earnings['pred'] - earnings['real']
    difference = initial_investment.append(earnings[['dif']])
    sum = difference.cumsum()
    hold_df  = initial_investment.append(earnings[['real']])
    hold = hold_df.cumsum()
    sum.columns = ['Bitcoin earnings']
    sum['No investment'] = investment
    sum['Hold investment'] = hold
    st.markdown("""### Making money!""")
    st.write('These are your earnings if you used our model against just saving it!')
    st.line_chart(sum)
    last_money = sum.iloc[-1:,:]
    extra_cash = last_money['Bitcoin earnings'] - last_money['No investment']
    st.write(f"If you used our model instead of just saving it you woul've made an extra: ${extra_cash.values[0]}")

def app():
    st.markdown("""# Past performance """)

    st.write(' ')
    st.write('In this section you will be able to see how much money you would have made if you used our model in the past instead of just saving it!')

    # Ask for innitial investment:
    st.markdown("""### Let's Invest some money!""")
    investment = st.slider('How much money are we investing?', 100, 1000, 10)
    st.write(' ')
    st.write('Select a time period to invest!')
    min_start = datetime.datetime(2021, 5, 3)
    max_value = datetime.datetime(2021, 9, 7)
    start_date = st.date_input('Start date', min_start, min_value = min_start, max_value = max_value)
    end_date = st.date_input('End date', max_value, min_value  = min_start+datetime.timedelta(days=1), max_value = max_value)

    if start_date < end_date:
        #st.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
        past_performance = get_past_data(start_date, end_date)
        plot_bitcoin_change(past_performance)
        get_earnings(investment, past_performance, start_date)
    else:
        st.error('Error: End date must fall after start date.')
