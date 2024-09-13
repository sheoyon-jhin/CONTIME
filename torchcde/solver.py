import torch
import torchdiffeq

import warnings
import numpy as np 

class _ContinuousDelayField(torch.nn.Module):
    def __init__(self, X, func,device):
        super(_ContinuousDelayField, self).__init__()
        
        self.X = X
        self.func = func
        
    
    def forward(self, t, inputz):
        
        h,z= inputz # h : 256,49 z : 256,55
        
        vector_field,out = self.func(t,z,self.X) 

        return (vector_field,out)

    
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(-1)


def contint_delay(X, func, z0,h0, t,device, adjoint=True, backend="torchdiffeq", **kwargs):
    
    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if adjoint:
        if "adjoint_atol" not in kwargs:
            kwargs["adjoint_atol"] = kwargs["atol"]
        if "adjoint_rtol" not in kwargs:
            kwargs["adjoint_rtol"] = kwargs["rtol"]
    
    if 'method' not in kwargs:
        kwargs['method'] = 'rk4'
    if kwargs['method'] == 'rk4':
        if 'options' not in kwargs:
            kwargs['options'] = {}
        options = kwargs['options']
        if 'step_size' not in options and 'grid_constructor' not in options:
            time_diffs = 1.0 # 0.5
            options['step_size'] = time_diffs
    
    vector_field = _ContinuousDelayField(X=X, func=func,device=device)
    if backend == "torchdiffeq":
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        out_h,out_z = odeint(func=vector_field, y0=(h0,z0), t=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    
    batch_dims = range(1, len(out_h.shape) - 1)
    out_h = out_h.permute(*batch_dims, 0, -1)
    batch_dims = range(1, len(out_z.shape) - 1)
    out_z = out_z.permute(*batch_dims, 0, -1)
    

    return out_h,out_z


