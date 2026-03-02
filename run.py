import argparse
import os
import sys
import torch 
import random
import numpy as np
from exp.Main import Exp_Main

#用于七个子数据集 

parser = argparse.ArgumentParser(description='ST-Mixer')

# random seed
parser.add_argument('--random_seed', type=int, default=2023, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')   
 
parser.add_argument('--model',  default='STMixer', help='STMixer')   
# data loader
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--data_path', type=str, default=r'.\all_six_datasets\traffic\traffic.csv', help='数据集路径')###
parser.add_argument('--features', type=str, default='M', help='features')
parser.add_argument('--data_set', type=str, default='traffic', help='traffic,electricity,weather,ETTh1,ETTh2,ETTm1,ETTm1') 
parser.add_argument('--time_data', type=str, default='linear', help='linear,embeding，决定生成的时间变量应使用什么方式进行嵌入，本项目仅可使用linear') 


# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length，在本项目中用不到')
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

parser.add_argument('--decomp_len', type=int, default=336+24*4, help='用于时间序列分解的长度')
# model define 
parser.add_argument('--input_size', type=int, default=862, help='x size')  ###
parser.add_argument('--mark_size', type=int, default=5, help='x_mark size') ###
parser.add_argument('--output_size', type=int, default=862, help='output size')  ###
 
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed_size', type=int, default=32, help='趋势项卷积嵌入的隐藏层维度')
parser.add_argument('--sample_embed_size', type=int, default=32, help='下采样最后一层和协变量融合时的MLP隐藏层维度')
parser.add_argument('--time_embed_size', type=int, default=16, help='趋势项和最终项进行协变量融合时的MLP隐藏层维度')
parser.add_argument('--sample_fusion', type=int, default=4, help='下采样融合的维度')
parser.add_argument('--singe_sample_dim', type=int, default=16, help='下采样时各个分支的隐藏层维度')

parser.add_argument('--t_size', type=int, default=8, help='时间协变量信息的嵌入维度')  
parser.add_argument('--coeff',type=float,  default=1, help='季节项下采样时，MLP中间层维度大小的系数')


parser.add_argument('--d_model', type=int, default=128, help='patchtst,patch嵌入维度')
parser.add_argument('--n_heads', type=int, default=4, help='patchtst,多头数量')
parser.add_argument('--d_ff', type=int, default=128*2, help='patchtst,前馈层大小')

parser.add_argument('--sampling_list',  type=int,nargs = '+',  default=[2,3,4,8], help='[1, 2, 3, 4, 6, 8, 12]，季节项下采样粒度')
parser.add_argument('--patch_len_list', type=int, default=5, help='趋势项卷积核大小')
parser.add_argument('--stride_list', type=int, default=2, help='趋势项卷积步幅大小')
parser.add_argument('--layer_num_cnn', type=int, default=3, help='趋势项卷积的层数')

parser.add_argument('--elayer_num', type=int, default=1, help='季节项的层数，patchtst层数')
parser.add_argument('--dlayer_num', type=int, default=2, help='趋势项的层数')

parser.add_argument('--norm', type=bool, default=True, help='ReVIN')   

parser.add_argument('--moving_avg',  type=int, nargs = '+',  default=[12,24,48], help='moving_avg') 

parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
#正则化
parser.add_argument('--weight_decay', type=float, default=0, help='正则项权重') 
parser.add_argument('--p', default=2, type = int, help='l1,l2正则化')

# optimization
parser.add_argument('--report_num', type=int, default=100, help='迭代多少次报告一次')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')  
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate') 
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type4', help='adjust learning rate') 
parser.add_argument('--lr_ratio', type=float, default=0.75, help='学习率衰减系数, lradj=type5时有效') 
parser.add_argument('--val_time', type=int, default=1, help='验证测试的此类') 

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu',  type=int,nargs = '+', default=[0], help='gpu')  

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

print('Args in experiment:')
data_parser = {
    'ETTh1':{'data':r'./all_six_datasets/ETT-small/ETTh1.csv' ,'M':[7,5,7],'S':[1,5,1] },
    'ETTh2':{'data':r'./all_six_datasets/ETT-small/ETTh2.csv' ,'M':[7,5,7],'S':[1,5,1] },
    'ETTm1':{'data':r'./all_six_datasets/ETT-small/ETTm1.csv' ,'M':[7,6,7],'S':[1,6,1] },
    'ETTm2':{'data':r'./all_six_datasets/ETT-small/ETTm2.csv' ,'M':[7,6,7],'S':[1,6,1] },
    'weather':{'data':r'./all_six_datasets/weather/weather.csv' ,'M':[21,6,21],'S':[1,6,1] },
    'electricity':{'data':r'./all_six_datasets/electricity/electricity.csv' ,'M':[321,7,321],'S':[1,7,1] },
    'traffic': {'data': r'./all_six_datasets/traffic/traffic.csv' , 'M': [862, 5, 862], 'S': [1, 5, 1] }, 
}
if args.data_set in data_parser.keys():
    print("开始进行参数匹配")
    data_info = data_parser[args.data_set]
    args.data_path = data_info['data'] 
    args.input_size, args.mark_size, args.output_size = data_info[args.features]
    print(args) 
    print("#"*100)
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = 'sl{}_ll{}_pl{}_mov{}_samp{}_data_set{}_Model{}_seed{}'.format(
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.moving_avg,
                args.sampling_list,
                args.data_set,
                args.model,
                args.random_seed)

        exp = Exp(args)  # set experiments
        print(exp.model)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        torch.cuda.empty_cache()
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = 'sl{}_ll{}_pl{}_mov{}_samp{}_data_set{}_Model{}_seed{}'.format(
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.moving_avg,
            args.sampling_list,
            args.data_set,
            args.model,
            args.random_seed)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
