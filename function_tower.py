
import torch 
import numpy as np 
global dhdt_list 
global t_list 
dhdt_list = torch.Tensor()
t_list = []

class ContGruFunc_Delay(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels,file_path,rnd,time_max):
        super(ContGruFunc_Delay, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.W_r = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_z = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.W_h = torch.nn.Linear(input_channels, hidden_channels, bias=False)
        self.U_r = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_z = torch.nn.Linear(hidden_channels, hidden_channels)
        self.U_h = torch.nn.Linear(hidden_channels, hidden_channels)
        self.file = file_path
        self.rnd = rnd
        self.dhdt_list = dhdt_list 
        self.t_list = t_list
        self.time_max = time_max 
    def forward(self, t,x, h,dxdt):
        
        self.time = t.item()
        if t ==0:
            h_past = h
        else:
            
            h_past = torch.Tensor(np.load(self.file+"/h_past/h_past_"+str(self.rnd)+".npy")).to(h)
        
        r = self.W_r(x) + self.U_r(h_past)
        
        r = r.sigmoid()
        z = self.W_z(x) + self.U_z(h_past)
        z = z.sigmoid()
        g0 = self.W_h(x) + self.U_h(r * h_past)
        g = g0.tanh()
        h_ = torch.mul(z,h_past) + torch.mul((1-z),g) # save h at t 
         
        np.save(self.file+'/h_past/h_past_'+str(self.rnd)+'.npy',h_.cpu().detach().numpy())

        hg = h_past - g
        
        if t==0:
            dhpast_dt = (1 - z) * (g - h)
        else:
            
            dhpast_dt = torch.Tensor(np.load(self.file+"/dhpastdt/dhpastdt_"+str(self.rnd)+".npy")).to(h)
        
        control_gradient = dxdt.derivative(t)  # 256,28
        dAdt =((self.W_z.weight @ control_gradient.unsqueeze(-1)) + (self.U_z.weight@dhpast_dt.unsqueeze(-1))).squeeze(-1) # dAdt = 10,49,1
        dzdt =torch.mul(torch.mul(z,(1-z)),dAdt)
        drdt = torch.mul(torch.mul(r,(1-r)),((self.W_r.weight @ control_gradient.unsqueeze(-1))+(self.U_r.weight@dhpast_dt.unsqueeze(-1))).squeeze(-1)) #drdt : 10,49
        dBdt =(self.W_h.weight @ control_gradient.unsqueeze(-1)).squeeze(-1) + torch.mul((self.U_h.weight@drdt.unsqueeze(-1)).squeeze(-1),h) +torch.mul((self.U_h.weight@r.unsqueeze(-1)).squeeze(-1),dhpast_dt)
        dgdt = torch.mul(torch.mul((1-g),(1+g)),dBdt)
        dhgdt = dhpast_dt - dgdt
        
        dhdt = torch.mul(dzdt,hg) + torch.mul(z,dhgdt) + dgdt 
        np.save(self.file+'/dhpastdt/dhpastdt_'+str(self.rnd)+'.npy',dhdt.cpu().detach().numpy())
        
        if self.time%1 == 0 and self.time not in self.t_list:
            
            self.t_list.append(self.time)
            if self.dhdt_list.shape[0]>0:
                
                self.dhdt_list = torch.cat([self.dhdt_list,dhdt.unsqueeze(0)],dim=0)
                
            else:
                
                self.dhdt_list = dhdt.unsqueeze(0)
            if self.time_max - self.time <= 1: 
                
                np.save(self.file+'/dhpastdt/dhdt_'+str(self.rnd)+'.npy', self.dhdt_list.cpu().detach().numpy())
                # self.dhdt_list = torch.Tensor()
                # self.t_list = []
        else:
            
            if (self.time_max > self.time_max -1  and self.time == 0) or (self.time_max < self.time_max -1  and self.time ==1): 
                self.t_list = []
                
                self.dhdt_list = dhdt.unsqueeze(0) 
                self.dhdt_list = torch.Tensor()
        
        return dhdt


class ContinuousDelayRNNConverter(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, model):
        super(ContinuousDelayRNNConverter, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.model = model
        self.linear = torch.nn.Linear(self.hidden_channels,self.input_channels+self.hidden_channels)
        
    

    def forward(self,t, z,dxdt):
        
        x = z[..., : self.input_channels]
        h = z[..., self.input_channels :]
        h = h.clamp(-1, 1)
        
        model_out = self.model(t,x, h,dxdt) # 1024,49
        out = self.linear(model_out)
        
        return model_out,out


def GRU_ODE_Delay(input_channels, hidden_channels,file_path,rnd,time_max=None):
    
    func = ContGruFunc_Delay(input_channels=input_channels, hidden_channels=hidden_channels,file_path=file_path,rnd=rnd,time_max = time_max)
    return ContinuousDelayRNNConverter(input_channels=input_channels,
                                            hidden_channels=hidden_channels,
                                            model=func)


