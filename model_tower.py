import torch
import torchcde
import numpy as np
import torchdiffeq


CUBICS = ['natural_cubic','cubic']

from sklearn.preprocessing import StandardScaler,MinMaxScaler

class moving_avg(torch.nn.Module):
    
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if len(x.shape)>2:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        else:
            
            front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
            end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x)
            
        return x
class series_decomp(torch.nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class CONTime(torch.nn.Module):
    def __init__(self,func,args, input_channels, hidden_channels, output_channels,pred_len,device,file_path,rnd,alpha,beta, interpolation="cubic"):
        super(CONTime, self).__init__()
        self.interpolation = interpolation
        self.func = func
        self.func2 = func
        input_channels = input_channels
        
        if self.interpolation =='linear':
            input_channels = input_channels*4
        
        self.readout = torch.nn.Linear(hidden_channels,input_channels)
        
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.rev_initial=torch.nn.Linear(input_channels,hidden_channels)
        self.get_prob = torch.nn.Linear(1,2)
        self.input_channels = input_channels
        self.device=device
        self.pred_len = pred_len
        self.file =file_path
        
        self.alpha = alpha
        self.beta = beta
        kernel_size = 25
        self.rnd = rnd
        
        self.decompsition = series_decomp(kernel_size)
        
    def forward(self,args,x,coeffs,times):
        if self.interpolation  in CUBICS:
            X = torchcde.CubicSpline(coeffs) 
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)  # torch.Size([256, 28])
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")
        seasonal_init, trend_init = self.decompsition(x)
        X0 = X.evaluate(X.interval[0]) 
        
        batch_dims = coeffs.shape[:-2]
        rev_X0 = X.evaluate(X.interval[-1])
        
        if len(X0.shape)>3:
            X0=X0.squeeze(-1)
            rev_X0.squeeze(-1)
        z0 = self.initial(X0)
        rev_z0 = self.rev_initial(rev_X0)
        
        
        z0_extra = torch.zeros(
                    *batch_dims, self.input_channels, dtype=z0.dtype, device=z0.device
                )
        h0 = z0 
        rev_h0 = rev_z0
        z0 = torch.cat([z0_extra, z0], dim=-1) 
        rev_z0 = torch.cat([z0_extra,rev_z0],dim=-1)
        
        h_T,z_T = torchcde.contint_delay(X=X,
                                z0=z0,
                                h0=h0,
                                func=self.func,
                                t=times,
                                device=self.device)
        rev_times = torch.flip(times,dims=[0])
        
        rev_h_T,rev_z_T = torchcde.contint_delay(X=X,
                                z0=rev_z0,
                                h0=rev_h0,
                                func=self.func2,
                                t=rev_times,
                                device=self.device)
        
        rev_h_T = torch.flip(rev_h_T,dims=[1])
        rev_z_T = torch.flip(rev_z_T,dims=[1])
        h_T = h_T + rev_h_T  
        z_T = z_T + rev_z_T 
        seq_len = times.shape[0]
        dhdt = torch.Tensor(np.load(self.file+"/dhpastdt/dhdt_"+str(self.rnd)+".npy")).to(h_T).permute(1,0,2)
        
        
        h_T =  h_T[:,seq_len-self.pred_len:,:]
        
        trend_init  = trend_init[:,seq_len-self.pred_len:,:]
        
        pred_y = self.readout(h_T) 
        weights = self.readout.weight.permute(1,0)
        dyhat_dt = torch.matmul(dhdt,weights)
        dyhat_dt =  dyhat_dt[:,-self.pred_len+1:,:]
        
        return pred_y,dyhat_dt
    
    
