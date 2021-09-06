import pandas as pd
from pmaw import PushshiftAPI
from datetime import datetime
from tqdm.notebook import tqdm
from sklearn.impute import SimpleImputer
import numpy as np
import string 
import datetime
import transformers
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from reddit_scrape import scrapping_posts, count_daily_posts

# Remove punctuation:
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.str.replace(punctuation, '') 
    return text

# All lower case:
def lower_all(text):
    return text.str.lower()

# Remove numbers:
def remove_numbers(text):
    return text.str.replace('\d+', '')

def preprocesing(df, col_name):
    # Remove punctuation:
    df['clean_text']  = remove_punctuation(df[col_name])
    # All lower case:
    df['clean_text'] = lower_all(df['clean_text'])
    # Remove numbers:
    df['clean_text'] = remove_numbers(df['clean_text'])
    return df

# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def load_model():
    # Load tokenizer and model, create trainer
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)
    return tokenizer, trainer, model

def create_preddiction(pred_texts):
    tokenizer, trainer, model = load_model()
    tokenized_texts = tokenizer(pred_texts,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)
    # Run predictions
    predictions = trainer.predict(pred_dataset)
    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)
    # Create a dataframe to return:
    return pd.DataFrame(list(zip(pred_texts,preds,labels,scores)), columns=['text','pred','label','score'])

def create_samples(df, start, end):
    #print(f'create_samples enter shape {df.shape}')
    
    n_samples = 20
    df_sample_list = []
    dt_range = pd.date_range(start = start, end = end, freq='d')
    for dt in pd.date_range(start = start, end = end, freq='d'):
        # print(f"df_date = {df['date'] }")
        # print(f"df[df['date'] == dt] = {df[df['date'] == dt]}")
        df_almost_sample = df[df['date'] == dt].copy()
        # print(f'df_almost_sample = {df_almost_sample}')
        if len(df_almost_sample) > n_samples:
            df_sample = df_almost_sample.sample(n_samples)
        else:
            df_sample = df_almost_sample
        df_sample_list.append(df_sample)
    df_all_sample = pd.concat(df_sample_list, axis = 0)
    return df_all_sample

def merge_sentiment_date(df_sentiment_analysis, df_useful, count_df):
    #df_useful = useful_only(df)
    df_sentiment_analysis = df_sentiment_analysis.merge(df_useful, how = 'inner').copy()
    df_sentiment_analysis['pred'].replace(0,-1, inplace = True) 
    df_sentiment_analysis['real_score'] = df_sentiment_analysis['pred']*df_sentiment_analysis['score']
    df_per_day = df_sentiment_analysis.groupby(['date']).mean()[['real_score']]
    count_df['date'] = pd.to_datetime(count_df['date']).copy()
    df_final = df_per_day.merge(count_df, on = 'date', how = 'inner')
    return df_final

def useful_only(df):
    df_useful = df[['clean_text','date']]
    df_useful.columns = ['text', 'date']
    df_useful['date'] = pd.to_datetime(df_useful['date'].dt.date).copy()
    return df_useful

def all_reddit(start, end):
    #scrapping:
    df = scrapping_posts(start, end)
    count_df = count_daily_posts (df)
    #all the other stuff
    df_preproc = preprocesing(df, 'title')
    #load_model()
    df_sample = create_samples(df_preproc, start, end)
    df_useful_only = useful_only(df_sample)
    df_sentiment_analysis = create_preddiction(list(df_useful_only['text']))
    df_final = merge_sentiment_date(df_sentiment_analysis, df_useful_only, count_df)
    df_final.to_csv('../raw_data/reddit_sentiment_n_posts.csv')
    return df_final

if __name__ == '__main__':
    start = datetime.datetime(2021, 9, 4)
    end = datetime.datetime(2021, 9, 5)
    all_reddit(start, end)

    



