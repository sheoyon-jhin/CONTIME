
import math
import torch
import torchcde
import datasets
import sklearn.model_selection
import torchdiffeq
from random import SystemRandom
import random
import numpy as np 
from parse import parse_args
import pathlib
import os 
import time 
import tqdm
import contime_dataset
import control_tower
import warnings
warnings.filterwarnings(action='ignore')

from tslearn.metrics import dtw, dtw_path
args = parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    
    return mae, mse, rmse, mape, mspe, rse, corr



def evaluate(epoch,model,optimizer,dataloader,model_name,times,loss_fn,NLL_fn):
    model.eval()
    total_dataset_size = 0
    total_mse= [] 
    total_dtw = []
    total_dt = []
    total_loss = []
    total_tdi=[]
    
    for batch in dataloader:
        loss_tdi_f=[]
        loss_dtw_f = []
        
        batch_coeffs,batch_x, batch_y = batch
        pred_y,dydt = model(args,batch_x,batch_coeffs,times)
                
        mse = loss_fn(pred_y,batch_y)
        
        batch_y_epi = batch_y[:,1:,:]
        batch_y_pre = batch_y[:,:-1,:]
        batch_y_diff = batch_y_epi - batch_y_pre
        dt_loss = loss_fn(dydt,batch_y_diff)
        batch_size = batch_y.shape[0]
        features = batch_y.shape[-1]
        loss = (args.alpha * mse) + (args.beta * dt_loss)
            
        for f in range(features):
            loss_tdi_ = 0
            loss_dtw_ = 0 
            for k in range(batch_size):         
                target_k_cpu = batch_y[k,:,f].view(-1).detach().cpu().numpy()
                output_k_cpu = pred_y[k,:,f].view(-1).detach().cpu().numpy()

                path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                loss_dtw_ += sim
                        
                Dist = 0
                for i,j in path:
                        Dist += (i-j)*(i-j)
                loss_tdi_ += Dist / (args.pred_len*args.pred_len)      
            loss_tdi_f.append(loss_tdi_/batch_size)   
            loss_dtw_f.append(loss_dtw_/batch_size)   
            
        loss_dtw = np.average(loss_dtw_f)
        loss_tdi = np.average(loss_tdi_f)
        b_size = batch_y.size(0)
        
        total_dataset_size +=b_size
        total_mse.append(mse.item()) 
        total_dt.append(dt_loss.item())
        total_dtw.append(loss_dtw)
        total_tdi.append(loss_tdi)
        total_loss.append(loss.item())
    total_mse =np.average(total_mse)    
    total_dtw = np.average(total_dtw)
    total_dt = np.average(total_dt)
    total_loss = np.average(total_loss)
    total_tdi = np.average(total_tdi)
    return total_mse,total_dtw,total_dt,total_tdi,total_loss
    
        
def load_model(args,model_path,visualize_version='test'): 
    device="cuda"
    model_name = args.model
    
    
    train_dataloader, val_dataloader,test_dataloader,input_channels ,output_channels= contime_dataset.get_dataset(args,device,visualization=True)
    model = control_tower.Model_selection_part(args,input_channels=input_channels,output_channels=output_channels, device=device )
    
    times = torch.Tensor(np.arange(args.seq_len))
    model=model.to(device)
    times=times.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    
    loss_fn = torch.nn.MSELoss()
    NLL_fn = torch.nn.NLLLoss()
    ckpt_file = model_path
    
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    breaking=False
    
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None   
    for epoch in range(1):
        if breaking:
            break
        model.eval()
        
        total_dataset_size = 0
        full_pred_y = torch.Tensor()
        full_true_y =torch.Tensor()
        full_x = torch.Tensor()
        loss_dtw = [] 
        loss_tdi = []
        preds = [] 
        trues = [] 
        if visualize_version=='train':
            dataloader = train_dataloader
        elif visualize_version=='val':
            dataloader = val_dataloader  
        else: 
            dataloader = test_dataloader  
        
        for batch in dataloader:
            loss_tdi_f = []
            loss_dtw_f =[]
            batch_coeffs,batch_x, batch_y = batch
            
            if breaking:
                break
            
            pred_y,pred_prob = model(args,batch_x,batch_coeffs,times)
            
            b_size = batch_y.size(0)
            
            mse = loss_fn(pred_y,batch_y)
            pred = pred_y.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            batch_y_epi = batch_y[:,1:,:]
            batch_y_pre = batch_y[:,:-1,:]
            batch_y_diff = batch_y_epi - batch_y_pre
            true_prob = (batch_y_diff>0).to(batch_y.dtype) 
            pred_prob = pred_prob.to(batch_y.dtype)
            batch_size = batch_y.shape[0]
            features = batch_y.shape[-1]
            for f in range(features):
                loss_tdi_ = 0
                loss_dtw_ = 0 
                for k in range(batch_size):         
                    target_k_cpu = batch_y[k,:,f].view(-1).detach().cpu().numpy()
                    output_k_cpu = pred_y[k,:,f].view(-1).detach().cpu().numpy()

                    path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                    loss_dtw_ += sim
                            
                    Dist = 0
                    for i,j in path:
                            Dist += (i-j)*(i-j)
                    loss_tdi_ += Dist / (args.pred_len*args.pred_len)      
                loss_tdi_f.append(loss_tdi_/batch_size)   
                loss_dtw_f.append(loss_dtw_/batch_size)   
               
            loss_dtw = np.average(loss_dtw_f)
            loss_tdi = np.average(loss_tdi_f)
            preds.append(pred)
            trues.append(true)
             
            full_pred_y=torch.cat([full_pred_y,pred_y.squeeze(-1).cpu()],dim=0)
            full_true_y=torch.cat([full_true_y,batch_y.squeeze(-1).cpu()],dim=0)
            full_x=torch.cat([full_x,batch_x.cpu()],dim=0)
            
            optimizer.zero_grad()
            total_dataset_size += b_size
            
        preds = np.array(preds)
        trues = np.array(trues)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        return mae, mse, rmse, mape, mspe, rse, corr, np.average(loss_tdi),np.average(loss_dtw)
        
def train(args,model,times,train_dataloader, val_dataloader,test_dataloader,optimizer,loss_fn,NLL_fn,device):
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None 
    num_epochs=args.epoch
    breaking=False
    tqdm_range = tqdm.tqdm(range(num_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    best_loss  = np.inf
    for epoch in tqdm_range:
        
        if breaking:
            break
        model.train()
        
        start_time= time.time()
        total_dataset_size = 0
        train_mse=[]
        
        train_dt = [] 
        train_dtw = []
        
        train_tdi = []
        best_train_mse = np.inf
        loss_dtw = []
        loss_tdi = []
        for batch in train_dataloader:
            loss_tdi_f=[]
            loss_dtw_f =[]
            if breaking:
                break
            batch_coeffs,batch_x, batch_y = batch
            pred_y,dydt = model(args,batch_x,batch_coeffs,times)
            
            
            mse = loss_fn(pred_y,batch_y)
            batch_y_epi = batch_y[:,1:,:]
            batch_y_pre = batch_y[:,:-1,:]
            batch_y_diff = batch_y_epi - batch_y_pre
            
            
            loss_dtw, loss_tdi = 0,0
            batch_size = batch_y.shape[0]
            features = batch_y.shape[-1]
            for f in range(features):
                loss_tdi_ = 0
                loss_dtw_ = 0 
                for k in range(batch_size):         
                    target_k_cpu = batch_y[k,:,f].view(-1).detach().cpu().numpy()
                    
                    output_k_cpu = pred_y[k,:,f].view(-1).detach().cpu().numpy()
                    
                    path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                    loss_dtw_ += sim
                            
                    Dist = 0
                    for i,j in path:
                            Dist += (i-j)*(i-j)
                    loss_tdi_ += Dist / (args.pred_len*args.pred_len)      
                loss_tdi_f.append(loss_tdi_/batch_size)   
                loss_dtw_f.append(loss_dtw_/batch_size)   
               
            loss_dtw = np.average(loss_dtw_f)
            loss_tdi = np.average(loss_tdi_f)
            dt_loss = loss_fn(dydt,batch_y_diff)
            
            if np.isnan(mse.item()):
                breaking = True
            loss = (args.alpha * mse) + (args.beta * dt_loss)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            b_size = batch_y.size(0)
            
            total_dataset_size += b_size
            train_mse.append(mse.item())
            train_dt.append(dt_loss.item())
            train_dtw.append( loss_dtw )
            train_tdi.append( loss_tdi )
        train_mse = np.average(train_mse)    
        train_dtw = np.average(train_dtw)
        
        train_dt = np.average(train_dt)
        train_tdi = np.average(train_tdi)
        if train_mse * 1.0001 < best_train_mse:
            best_train_mse = train_mse
            
        
        print('Epoch: {}  Train MSE: {:.4f}, Train DTW : {:.4f}, Train dT : {:.4f} Train TDI: {:.4f} Time :{:.4f}'.format(epoch,train_mse,train_dtw,train_dt,train_tdi,(time.time()-start_time)))
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        
        val_mse,val_dtw,val_dt,val_tdi,val_loss = evaluate(epoch,model,optimizer,val_dataloader,args.model,times,loss_fn,NLL_fn)
        test_mse,test_dtw,test_dt,test_tdi ,test_loss= evaluate(epoch,model,optimizer,test_dataloader,args.model,times,loss_fn,NLL_fn)
        
        print('Epoch: {}   Validation MSE: {:.4f}, Validation DTW : {:.4f}, Validation dT : {:.4f} TDI: {:.4f} Time :{:.4f}'.format(epoch, val_mse,val_dtw,val_dt,val_tdi,(time.time()-start_time)))
        print('Epoch: {}   Test MSE: {:.4f}, Test DTW : {:.4f}, Test dT: {:.4f} TestTDI: {:.4f} Time :{:.4f}'.format(epoch, test_mse,test_dtw,test_dt,test_tdi,(time.time()-start_time)))
        print(f"memory_usage:{memory_usage}")

    
def main(model_name=args.model,num_epochs=args.epoch):
    
    manual_seed = args.seed
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)
    print(f"Setting of this Experiments {args}")
    device="cuda"
    
    train_dataloader, val_dataloader,test_dataloader,input_channels ,output_channels= contime_dataset.get_dataset(args,device)
    model = control_tower.Model_selection_part(args,input_channels=input_channels,output_channels=output_channels, device=device )
    times = torch.Tensor(np.arange(args.seq_len))
    if args.pretrained:
        load_model(args)
        exit()
    
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    NLL_fn = torch.nn.NLLLoss()
    if args.training:  
        
        ckpt_file = train(args,model,times,train_dataloader, val_dataloader,test_dataloader,optimizer,loss_fn,NLL_fn,device)
    else:
        MODEL_PATH = '../CONTIME_KDD/trained_model/'+str(args.model)+'/'+str(args.dataset)+'/'+str(args.seq_len)+"_"+str(args.pred_len)+"_"+str(args.stride_len)+"_"+str(args.note)+"_"+str(args.lr)+"_"+str(args.alpha)+"_"+str(args.beta)+"/"
        ckpt_file = MODEL_PATH+"contime.pth" 
        print("============> Evaluation <============")
        mae, mse, rmse, mape, mspe, rse, corr,tdi,dtw = load_model(args,ckpt_file,visualize_version=args.visualize_version)
        
        print("Final Results MAE: {:.4f} MSE: {:.4F} RMSE: {:.4f} MAPE: {:.4f} MSPE: {:.4f} RSE: {:.4f} TDI: {:.4f} DTW: {:.4f}".format(mae,mse,rmse,mape,mspe,rse,tdi,dtw))
        

if __name__ == '__main__':
    main()
    
