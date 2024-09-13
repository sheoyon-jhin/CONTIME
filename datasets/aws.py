import collections as co
import numpy as np
import os
import pathlib
import pandas as pd 
import torch
import urllib.request
import zipfile

from . import common
from sklearn.preprocessing import StandardScaler,MinMaxScaler
here = pathlib.Path(__file__).resolve().parent

DATA_PATH = os.path.dirname(os.path.abspath(__file__))

def _pad(channel, maxlen):
    
    channel = torch.tensor(channel) 
    out = torch.full((maxlen,), channel[-1]) 
    out[:channel.size(0)] = channel 
    return out 


def _process_data(missing_rate,look_window,forecast_window,stride_window,SAVE_PATH):
    PATH = os.path.dirname(os.path.abspath(__file__))
    
    torch.__version__
    scaler = StandardScaler()
    X_times = pd.read_csv(PATH + '/data/AMZN/AMZN.csv')
    X_times = X_times.drop(columns=['Date'])
    X_train = X_times.iloc[:int(X_times.shape[0]*0.7),:]
    scaler.fit(X_train.values)
    X_times = scaler.transform(X_times.values)
    X_times = np.array(X_times)
    
    
    # X_times = X_times[::-1]
    timelen = X_times.shape[0]
    X_seq,Y_seq =  common.get_sequences(X_times,timelen,look_window,forecast_window,stride_window)
    if missing_rate >0:
        X_seq = common.create_irregular(X_seq,missing_rate)
    train_X,train_Y = common.split_data(X_seq,Y_seq,flag='train')
    val_X,val_Y = common.split_data(X_seq,Y_seq,flag='val')
    test_X,test_Y = common.split_data(X_seq,Y_seq,flag='test')
    
    common.saved_preprocessed_data(SAVE_PATH,train_X,train_Y,flag='train')
    common.saved_preprocessed_data(SAVE_PATH,val_X,val_Y,flag='val')
    common.saved_preprocessed_data(SAVE_PATH,test_X,test_Y,flag='test')
    

def get_data(missing_rate,look_window,forecast_window,stride_window,note):
    base_base_loc = here / 'processed_data'
    loc = base_base_loc / ('aws_'+str(note)+'_seq_'+str(look_window)+'_pred_'+str(forecast_window)+'_stride_'+str(stride_window)+'_missing_'+str(missing_rate))
    SAVE_PATH = '/processed_data/aws_'+str(note)+'_seq_'+str(look_window)+'_pred_'+str(forecast_window)+'_stride_'+str(stride_window)+'_missing_'+str(missing_rate)
    PATH = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(loc):
        loc_list = os.listdir(loc)
        if len(loc_list)<6:
            _process_data(missing_rate,look_window,forecast_window,stride_window,SAVE_PATH)
        else:
            X_times = pd.read_csv(PATH + '/data/AMZN/AMZN.csv')
            X_times = X_times.drop(columns=['Date'])
               
    else:
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        _process_data(missing_rate,look_window,forecast_window,stride_window,SAVE_PATH)
    return loc