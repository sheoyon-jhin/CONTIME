import argparse

# argparser -> model name, hidden_channels, etc
# main(intensity = True, model_name='ncde',hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4)

def parse_args():
    parser = argparse.ArgumentParser(description='CONTIME')
    parser.add_argument('--seed', type=int, default=2021,help='Seed - Test your luck!')   
    parser.add_argument('--intensity', type=bool, default=True,help='Intensity')
    parser.add_argument('--model', type=str, default='contime',help='Model Name')
    parser.add_argument('--h_channels', type=int, default=49,help='Hidden Channels') 
    parser.add_argument('--lr', type=float, default=0.001,help='Learning Rate') 
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='Weight Decay') 
    parser.add_argument('--epoch', type=int, default=100,help='Epoch') 
    parser.add_argument('--batch', type=int, default=256,help='Batch size') 
    parser.add_argument('--data_name',type=str,default='AMZN')
    
    parser.add_argument('--solver_method', type=str, default='RK4',help='ODE Solver Methods') 
    parser.add_argument('--task', type=str, default='forecasting',help='Task') 
    
    parser.add_argument('--note',type=str,default=2023,help='date')
    
    parser.add_argument('--n_classes', type=int, default=2,help='number of class in classification') 
    
    #forecasting
    parser.add_argument('--seq_len',type=int, default = 20,help='look_window')
    parser.add_argument('--pred_len',type=int,default=10,help='forecast window')
    parser.add_argument('--stride_len',type=int,default=1,help='stride window')
    
    parser.add_argument('--dataset',type=str)
    #random missing
    parser.add_argument('--missing_rate',type=float,default=0.0,help='missing rate')
    parser.add_argument('--interpolation',type=str,default='cubic',help='Interpolation Method')
    
    parser.add_argument('--model_path',type=str)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--get_source', type=bool, default=False)
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--model_saved', type=bool, default=False)
    parser.add_argument('--visualize_version',type=str,choices=['train','val','test'],default='test')
    parser.add_argument('--training',type=bool, default=False)
    
    #loss_function
    parser.add_argument('--alpha',type=float,default=0.9,help='coefficient of mse')
    parser.add_argument('--beta',type=float,default=0.1,help='coefficient of L_Delta t')
    
    
    return parser.parse_args()
