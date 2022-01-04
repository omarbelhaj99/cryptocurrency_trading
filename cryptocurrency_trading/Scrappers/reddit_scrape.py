import pandas as pd
from pmaw import PushshiftAPI
from datetime import datetime
from tqdm.notebook import tqdm
import datetime
from sklearn.impute import SimpleImputer
import numpy as np

def scrapping_posts(start_date, end_date):
    # date eange
    end_minus1 =  end_date - datetime.timedelta(days=1)
    dt_range = pd.date_range(start = start_date, end = end_date, freq='d')
    api = PushshiftAPI()
    df_list = []
    pass_counter = 0
    for idx, date in tqdm(enumerate(dt_range)):
        if idx +1>= len(dt_range):
            break
        # start date
        after = int(dt_range[idx].timestamp())
        # end date
        before = int(dt_range[idx+1].timestamp())
        subreddit="bitcoin"
        limit=500
        comments = api.search_submissions(subreddit=subreddit, limit=limit, before=before, after=after)
        print(f'Retrieved {len(comments)} comments from Pushshift--{dt_range[idx].strftime("%Y-%m-%d"), dt_range[idx+1].strftime("%Y-%m-%d")}')
        try:
            df_temp = pd.DataFrame(comments)
            df_list.append(df_temp)
        except: 
            pass_counter+=1
            print(f"passing {pass_counter} - {dt_range[idx].strftime('%Y-%m-%d'), dt_range[idx+1].strftime('%Y-%m-%d')}")
    df = pd.concat(df_list, axis = 0)
    df_concat = concat_tittle_comment(df)
    df_datetime = converting_datetime(df_concat)
    return df_datetime

def converting_datetime(df):
    df['date'] = df.created_utc.apply(lambda d: datetime.datetime.fromtimestamp(int(d)))
    df['date'] = pd.to_datetime(df['date'].dt.date).copy()
    return df

def concat_title_comment(df):
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = '')
    imputer2 = SimpleImputer(missing_values='[removed]', strategy='constant', fill_value = '')
    imputer3 = SimpleImputer(missing_values='[deleted]', strategy='constant', fill_value = '')
    df['self_text']= imputer.fit_transform(df[['selftext']])
    df['self_text']= imputer2.fit_transform(df[['self_text']])
    df['self_text']= imputer3.fit_transform(df[['self_text']])
    df['concat'] = df['title']+df['self_text']
    return df


def count_daily_posts(df):
    count_df_all = df.groupby([df['date'].dt.date]).count()
    count_df = count_df_all[['date']]
    count_df.columns=['post_per_day']
    coms_per_day = df.groupby([df['date'].dt.date]).sum()[['num_comments']]
    count_df.merge(coms_per_day, on = 'date')
    count_df['date'] = count_df.index
    count_df.index =count_df.index.rename('index_date')
    return count_df

