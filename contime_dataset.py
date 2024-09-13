
import datasets
import pathlib
import torch
import os 
import torchcde
import numpy as np 
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import pandas as pd 
def inverse_transform(args,data):
    PATH = os.path.dirname(os.path.abspath(__file__))
    
    scaler = StandardScaler()
    X_times = pd.read_csv(PATH + '/datasets/data/'+str(args.data_name)+'/'+str(args.data_name)+'.csv')
    X_times = X_times.drop(columns=['Date'])
    X_train = X_times.iloc[:int(X_times.shape[0]*0.7),:]
    scaler.fit(X_train.values)
    data = data.reshape((-1,data.shape[-1]))
    return scaler.inverse_transform(data)


def get_dataset(args,device,visualization=False):
    
    seq_len = args.seq_len 
    pred_len=args.pred_len
    stride_len = args.stride_len
    missing_rate = args.missing_rate
    batch_size = args.batch
    time_intensity=True
    static_intensity=False
    
    note=args.note
    if args.dataset =='AWS':
        loc = datasets.aws.get_data(missing_rate,seq_len,pred_len,stride_len,note)
    here = pathlib.Path(__file__).resolve().parent
    base_base_loc = here / 'datasets/processed_data'
    
    
    if args.interpolation=='natural_cubic':
        coeff_loc = loc / ('NaturalCoeffs')
    else:    
        coeff_loc = loc / ('Coeffs')
    train_y       =  torch.load(str(loc) +'/train_Y.pt').unsqueeze(-1)
    val_y       =  torch.load(str(loc) +'/val_Y.pt').unsqueeze(-1)
    test_y       =  torch.load(str(loc) +'/test_Y.pt').unsqueeze(-1)
    
    
    train_X = torch.load(str(loc) +'/train_X.pt')
    val_X = torch.load(str(loc) +'/val_X.pt')
    test_X = torch.load(str(loc) +'/test_X.pt')
    
    if len(train_X.shape) != 3 :
        train_X = train_X.unsqueeze(-1)
        val_X = val_X.unsqueeze(-1)
        test_X= test_X.unsqueeze(-1)
        
    
    if static_intensity:
        input_channels = train_X.shape[-1]
    else:
        input_channels = train_X.shape[-1]
    output_channels = input_channels 
    
        
    
    if os.path.exists(coeff_loc):
        
        pass
    else:
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(coeff_loc):
            os.mkdir(coeff_loc)
        if not os.path.exists(coeff_loc):
            os.mkdir(coeff_loc)
        
        
        if args.interpolation =='natural_cubic':
            print("Start extrapolation!")
            
            train_coeffs = torchcde.natural_cubic_coeffs(train_X)
            
            torch.save(train_coeffs,str(coeff_loc)+'/train_coeffs.pt')
            print("finish extrapolation Train coeff")
            val_coeffs = torchcde.natural_cubic_coeffs(val_X)
            
            
            torch.save(val_coeffs,str(coeff_loc)+'/val_coeffs.pt')
            print("finish extrapolation Val coeff")
            test_coeffs = torchcde.natural_cubic_coeffs(test_X)
            torch.save(test_coeffs,str(coeff_loc)+'/test_coeffs.pt')
            print("finish extrapolation Test coeff")
            print("success!")
        else:
            print("Start extrapolation!")  
            train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)
            torch.save(train_coeffs,str(coeff_loc)+'/train_coeffs.pt')
            print("finish extrapolation Train coeff")
            val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(val_X)
            torch.save(val_coeffs,str(coeff_loc)+'/val_coeffs.pt')
            print("finish extrapolation Val coeff")
            test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
            torch.save(test_coeffs,str(coeff_loc)+'/test_coeffs.pt')
            print("finish extrapolation Test coeff")
            print("success!")
    
    
    train_coeffs = torch.load(str(coeff_loc)+'/train_coeffs.pt')
    val_coeffs = torch.load(str(coeff_loc)+'/val_coeffs.pt')
    test_coeffs = torch.load(str(coeff_loc)+'/test_coeffs.pt')
    
    
    train_coeffs=train_coeffs.to(device)
    val_coeffs=val_coeffs.to(device)
    test_coeffs=test_coeffs.to(device)
    train_X = train_X.to(device)
    val_X = val_X.to(device)
    test_X = test_X.to(device)
    train_y = torch.Tensor(np.nan_to_num(np.array(train_y.cpu())))
    train_y = train_y.to(device)
    val_y = torch.Tensor(np.nan_to_num(np.array(val_y.cpu())))
    val_y = val_y.to(device)
    test_y = torch.Tensor(np.nan_to_num(np.array(test_y.cpu())))
    test_y = test_y.to(device)
    if len(train_y.shape) !=3:
        train_y=  train_y.squeeze(-1)
        val_y=  val_y.squeeze(-1)
        test_y=  test_y.squeeze(-1)
    if args.task =='forecasting':
        output_channels = train_y.shape[-1]
    
    hidden_channels=args.h_channels
    if visualization or 'contime' in args.model:
        
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_X,train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)
        val_dataset = torch.utils.data.TensorDataset(val_coeffs, val_X,val_y)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_y.shape[0])
        test_dataset = torch.utils.data.TensorDataset(test_coeffs,test_X, test_y)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_y.shape[0])
    
    else:
        
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch,shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_coeffs, val_y)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_y.shape[0],shuffle=True)
        test_dataset = torch.utils.data.TensorDataset(test_coeffs, test_y)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_y.shape[0],shuffle=False)
    
    return train_dataloader,val_dataloader,test_dataloader,input_channels,output_channels 



