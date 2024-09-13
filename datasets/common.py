import os
import pathlib
import sklearn.model_selection
import sys
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

PATH = os.path.dirname(os.path.abspath(__file__))
def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)

def get_sequences(X,total_len,seq_len,pred_len,stride_len):
    X_seq = torch.Tensor()
    Y_seq = torch.Tensor()
    for _len in range(int((total_len-seq_len-pred_len+stride_len)/stride_len)):
        
        full_seq_temp = torch.Tensor([X[(_len*stride_len):(_len*stride_len)+seq_len]])
        full_y_seq_temp = torch.Tensor([X[(_len*stride_len)+seq_len:(_len*stride_len)+seq_len+pred_len]])
        
        X_seq=torch.cat([X_seq,full_seq_temp])
        Y_seq = torch.cat([Y_seq,full_y_seq_temp])
    return X_seq, Y_seq



def split_data(X,Y,flag ='train'):
    
    if flag=='train':
        X = X[:int(X.shape[0]*0.7)]
        Y = Y[:int(Y.shape[0]*0.7)]
    elif flag=='val':
        X = X[int(X.shape[0]*0.7):int(X.shape[0]*0.8)]
        Y = Y[int(Y.shape[0]*0.7):int(Y.shape[0]*0.8)]
    elif flag=='test':
        X = X[int(X.shape[0]*0.8):]
        Y = Y[int(Y.shape[0]*0.8):]
    
    
    return X,Y


def saved_preprocessed_data(SAVE_PATH,X,Y,flag='train'):
    
    torch.save(X,PATH+SAVE_PATH+'/'+str(flag)+'_X.pt')
    torch.save(Y,PATH+SAVE_PATH+'/'+str(flag)+'_Y.pt')
    print(f'{flag} preproceseed data just saved!')
    

