##imports
from sklearn.impute import KNNImputer
import pandas as pd
import os
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau






##impute
def imputer(df):
    imputer = KNNImputer(n_neighbors=2)
    df['reddit_sentiment'],df['reddit_post_count'],df['twitter_sentiment'] = imputer.fit_transform(df[['reddit_sentiment','reddit_post_count','twitter_sentiment']]).T
    return df

##train_test_split
def train_test_split(df):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]
    return train_df,val_df,test_df,num_features

def standardize(train_df,df):

    train_mean=train_df[['tweet_count','twitter_sentiment','volume','volatility','rsi','macd','reddit_sentiment','reddit_post_count']].mean()
    train_std=train_df[['tweet_count','twitter_sentiment','volume','volatility','rsi','macd','reddit_sentiment','reddit_post_count']].std()
    df[['tweet_count','twitter_sentiment','volume','volatility','rsi','macd','reddit_sentiment','reddit_post_count']]=(df[['tweet_count','twitter_sentiment','volume','volatility','rsi','macd','reddit_sentiment','reddit_post_count']]-train_mean)/train_std
    return df


##create target

def create_train_dataset(sequence_length,train_df):
    train_df['target'] = train_df['close_price'].shift(-sequence_length)
    train_df.drop('start', axis = 1, inplace = True)
    train_df.dropna()
    X_train = train_df.drop('target', axis = 1)
    y_train = train_df['target']
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_train, y_train, sequence_length=10, batch_size = 16)
    exx, exxy = next(iter(dataset))
    return dataset

def create_test_dataset(sequence_length,test_df):
    test_df['target'] = test_df['close_price'].shift(-sequence_length)
    X_test = test_df.drop('target', axis = 1)
    y_test = test_df['target']
    dataset_test = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_test, y_test, sequence_length=10)
    etx, ety = next(iter(dataset_test))
    return dataset_test

def create_val_dataset(sequence_length,val_df):
    val_df['target'] = val_df['close_price'].shift(-sequence_length)
    val_df.drop('start', axis = 1, inplace = True)
    X_val = val_df.drop('target', axis = 1)
    y_val =val_df['target']
    dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
    X_val, y_val, sequence_length=10)
    evx, evy = next(iter(dataset_val))
    return dataset_val


##train model

def model_setup():
    lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.GRU(50, return_sequences=True),
    tf.keras.layers.GRU(20),
    #tf.keras.layers.Dropout(0.2),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=20),

    tf.keras.layers.Dense(units=1)
    # Adding a second LSTM layer and some Dropout regularisation
    ])
    lstm_model.compile(optimizer = 'Adam', loss = 'mae')
    es = EarlyStopping(patience = 20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience = 15)
    lstm_model.fit(dataset, validation_data = dataset_val, epochs = 2_000, callbacks = [es, reduce_lr])



    ##save model? model performance???
    return lstm_model.evaluate(dataset_test)
