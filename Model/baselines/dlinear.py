# 出处：论文 Table 4/5 自跑 baseline，迁移自 part1 Model/Dlinear.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
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


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean

class Dlinear(nn.Module): 

    def __init__(self, args ):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Dlinear, self).__init__() 
        self.args = args
        self.seq_len = args.seq_len  
        self.pred_len = args.pred_len  
        self.decompsition = series_decomp_multi(args.moving_avg)  
 
        self.Linear_Seasonal = MLPBlock(args.seq_len  ,args.seq_len  ,args.pred_len,0,1  )
        self.Linear_Trend = MLPBlock(args.seq_len  ,args.seq_len  ,args.pred_len,0,1  )

        self.Linear_Seasonal.fc1.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.fc1.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
 
    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)   #（B,T,V）
        seasonal_init, trend_init = seasonal_init.permute( 0, 2, 1), trend_init.permute(0, 2, 1)#（B,V,T）
 
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output #（B,V,P）
        return x.permute(0, 2, 1) #（B,P,V）

 
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec ): 
        dec_out = self.encoder(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
 
#对季节项下采样
class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, dropout, mlp_mode=1) :
        super().__init__()
        self.mlp_mode = mlp_mode
        if mlp_mode==2:
            self.fc1 = nn.Linear(input_dim, mlp_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout) 
            self.fc2 = nn.Linear(mlp_dim, output_dim)
        elif mlp_mode==1:
            self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        if self.mlp_mode==2:
            return self.fc2(self.dropout(self.gelu(self.fc1(x))))
        elif self.mlp_mode==1:
            return self.fc1(x)