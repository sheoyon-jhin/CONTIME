# CONTIME
Addressing prediction delays in time series forecasting: A continuous GRU Approach with derivative regularization 
Newly accepted KDD 2024


# OTHER DATASETS WILL BE UPLOADED SOON!! (SORRY FOR THE LATE UPLOAD)


## Installation

```
pip install contime_kdd
conda activate contime_kdd
```



## Training code
```
python contime.py --dataset 'AWS' --batch 256 --task 'forecasting' --epoch 100 --model 'contime' --seq_len 104 --pred_len 24 --stride_len 1 --alpha 0.8  --lr 0.005 --beta 0.1 --seed 2021 --training 'True' --note '0126' --missing_rate 0 --data_name 'AMZN'
```


## Testing code 
```
python contime.py --dataset 'AWS' --batch 256 --task 'forecasting' --epoch 100 --model 'contime' --seq_len 104 --pred_len 24 --stride_len 1 --alpha 0.8  --lr 0.005 --beta 0.1 --seed 2021 --training '' --note '0126' --missing_rate 0 --data_name 'AMZN'
```

