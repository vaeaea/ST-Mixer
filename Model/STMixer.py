import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

############################################################################################################
#时间序列分解（论文 §4.1 Multi-scale Series Decomposition Module）
class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride, decomp_len, seq_len):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        self.end_padding = MLPBlock(self.decomp_len, self.decomp_len//2 , math.floor((self.kernel_size - 1) // 2), mlp_mode=2)
    def forward(self, x):
        end = self.end_padding( x.transpose(1,2)).transpose(1,2)
        self.end = end
        x = torch.cat([x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x, end

#单尺度分解（moving average，论文 §4.1）
class SingleScaleDecomposition(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, decomp_len, seq_len):
        super(SingleScaleDecomposition, self).__init__()
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.moving_avg = MovingAverage(kernel_size, 1, decomp_len, seq_len)

    def forward(self, x):
        moving_mean,end_padding= self.moving_avg(x) 
        res = x[:,-moving_mean.shape[1]:,:] - moving_mean
        return res[:,-self.seq_len:,:], moving_mean[:,-self.seq_len:,:],end_padding

#多尺度分解（论文 §4.1 Multi-scale Series Decomposition Module）
class MultiScaleSeriesDecomposition(nn.Module):
    """
    Multiple Series decomposition block 
    """
    def __init__(self, kernel_size, decomp_len, seq_len):
        super(MultiScaleSeriesDecomposition, self).__init__()
        self.kernel_size = kernel_size
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.series_decomp =  nn.ModuleList([SingleScaleDecomposition(kernel, decomp_len, seq_len) for kernel in kernel_size])
    def forward(self, x): 
        x = x
        moving_mean = []
        res = []
        end_paddings = []
        for func in self.series_decomp:
            sea, trend, end_padding = func(x)
            moving_mean.append(trend)
            res.append(sea)
            end_paddings.append(end_padding)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        self.sea = sea
        self.moving_mean = moving_mean
        return sea , moving_mean , end_paddings

############################################################################################################
#季节分支子模块（论文 §4.2 Seasonal Multi-downsampling Fusion Module）

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, dropout=0.1, mlp_mode=1) :
        super().__init__()
        self.mlp_mode = mlp_mode  #MLP多层还是单层
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
        
#单一粒度下采样模块（论文 §4.2 Downsampling Module，Figure 5）
class DownsamplingModule(nn.Module):
    def __init__(self, sample_fusion, sampling_list, current_layer, coeff, seq_len, pred_len, sampling, dropout, singe_sample_dim, mode='pre') :
        super().__init__()
        self.mode = mode
        
        assert sampling in [1, 2, 3, 4, 6, 8, 12, 24, 48]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sampling = sampling
        self.coeff = coeff
        self.current_layer = current_layer
        self.singe_sample_dim = singe_sample_dim
        if current_layer==0: #第一层和中间层的输入维度有区别，因此需要作出区分
            self.temporal_fac = MLPBlock(seq_len // sampling, int(coeff * seq_len * (1 + singe_sample_dim) // sampling // 2),
                         singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2)
        else:
            self.temporal_fac = MLPBlock(sample_fusion * seq_len // sampling,int( coeff * (sample_fusion+singe_sample_dim) * seq_len // sampling // 2) , singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2)
        
        if self.mode=='pre': #当位于最后一层时，需要额外映射为未来值
            self.predict = MLPBlock(seq_len, int(coeff * (seq_len*singe_sample_dim+pred_len) // 2)  , pred_len , dropout, mlp_mode=2) 
    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x): 
        if self.mode=='pre':
            x = x.transpose(1,2) #(B,C,T)
            x_to_trans = torch.stack([x[:, :, i::self.sampling] for i in range(self.sampling)],dim=1).view(x.shape[0], self.sampling, -1)  # (B,S,C*T/S)
            x = torch.reshape(torch.reshape(self.temporal_fac(x_to_trans), (x.shape[0], self.sampling, self.singe_sample_dim, self.seq_len // self.sampling)).permute(0,2,3,1), (x.shape[0], self.singe_sample_dim,-1) ).transpose(1,2) #(B,T,C)

            x = self.predict( x.transpose(1,2)).transpose(1,2)
            return x  #(B,T,C)

        elif self.mode=='trans':
            x = x.transpose(1,2) #(B,C,T)
            x_to_trans = torch.stack([x[:, :, i::self.sampling] for i in range(self.sampling)],dim=1).view(x.shape[0], self.sampling, -1)  # (B,S,C*T/S)
            x = torch.reshape(torch.reshape(self.temporal_fac(x_to_trans), (
            x.shape[0], self.sampling, self.singe_sample_dim, self.seq_len // self.sampling)).permute(0, 2, 3, 1),
                              (x.shape[0], self.singe_sample_dim, -1)).transpose(1, 2)  # (B,T,C)
            return x  #(B,T,C)            
            
            
            
#下采样融合层（论文 §4.2：multi-downsampling fusion module 的一层，merge 即 Parallel Fusion MLP_V）
class MultiDownsamplingFusionLayer(nn.Module):
    def __init__(self, sample_fusion, coeff, t_size, time_embed_size, embed_size, output_size, seq_len, pred_len, sampling_list, dropout, mode, singe_sample_dim, current_layer = None):
        super().__init__() 
        self.sampling_list = sampling_list      
        self.mode = mode
        self.coeff = coeff
        Fact_list = []
        for i in range(len(sampling_list)):
            Fact_list.append(DownsamplingModule(sample_fusion, sampling_list, current_layer, coeff, seq_len, pred_len, sampling_list[i], dropout, singe_sample_dim, mode))
        self.Fact_list = nn.ModuleList(Fact_list)
        
        if mode=='trans':
            self.merge = MLPBlock(len(sampling_list)*singe_sample_dim, int(self.coeff * ((len(sampling_list)*singe_sample_dim + sample_fusion) // 2)), sample_fusion, dropout, mlp_mode=2)
    def forward(self, x, x_mark):  #(B,T,d_model)   
        x_pre = []
        for i in range(len(self.sampling_list)):  #根据不同的下采样细粒度进行对数据的重新处理
            x_pre.append(self.Fact_list[i](x))
        
        if self.mode=='trans':
            x_pre = self.merge(    torch.cat(x_pre,-1)   ) #对不同下采样细粒度的处理结果进行合并
        return x_pre

#季节项处理部分（论文 §4.2 Seasonal Multi-downsampling Fusion Module，堆叠 M_s 层）
class SeasonalMultiDownsamplingFusion(nn.Module):
    def __init__(self, sample_fusion, coeff, layer_num, embed_size, output_size, seq_len, pred_len, sampling_list, dropout, time_embed_size, t_size,singe_sample_dim):
        super().__init__() 
        self.sampling_list = sampling_list      
        self.layer_num = layer_num   
        Fact_list = []
        if layer_num>1:
            for i in range(layer_num-1):
                Fact_list.append(MultiDownsamplingFusionLayer(sample_fusion, coeff, t_size, time_embed_size, embed_size, output_size, seq_len, pred_len, sampling_list, dropout, 'trans', singe_sample_dim,current_layer = i))  #trans:中间层不进行预测；current_layer=0 时输入维度不同，将影响到 DownsamplingModule 的输入维度。
            Fact_list.append(MultiDownsamplingFusionLayer(sample_fusion, coeff, t_size, time_embed_size, embed_size, output_size, seq_len, pred_len, sampling_list, dropout, 'pre',singe_sample_dim))  #pre:最后一层进行预测
            
        elif layer_num==1:
            Fact_list.append(MultiDownsamplingFusionLayer(sample_fusion, coeff, t_size, time_embed_size, embed_size, output_size, seq_len, pred_len, sampling_list, dropout, 'pre',singe_sample_dim, current_layer = 0))
        
        
        self.Fact_list = nn.ModuleList(Fact_list) 
    
    def forward(self, x, x_mark):  #(B,T,d_model)    
        for i in range(self.layer_num):   
            x = self.Fact_list[i](x, x_mark)
         
        return x  

    
###############################################################################################################    
#趋势分支 patching（论文 §4.3 Single-scale Patching）
class SingleScalePatching(nn.Module):
    def __init__(self, in_channels, out_channels, patch_len, stride, layer_num_cnn):
        super(SingleScalePatching, self).__init__()
 
        self.conv_patch = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_len, stride = stride),
                                        *[nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=patch_len, stride = stride)]*(layer_num_cnn-1))

    def forward(self, x): 
        x_patch = self.conv_patch(x.transpose(1,2)).transpose(1,2)
        return x_patch

#通过两组MLP进行局部信息patch内/patch间交互（论文 §4.3 Trend Information Extraction Module 内部结构）
class PatchMixing(nn.Module):
    def __init__(self, t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list, layer_num_cnn, mode='pre'):
        super(PatchMixing, self).__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list 
        self.pred_len = pred_len
        self.mode = mode
        mlp_len = seq_len
        for i in range(layer_num_cnn):
            mlp_len = (mlp_len-patch_len_list) // stride_list + 1
        self.temp_list = nn.Sequential(nn.Linear(mlp_len, mlp_len),nn.GELU(),nn.Linear(mlp_len, mlp_len))
        
        self.ff = MLPBlock(embed_size , 2*embed_size, embed_size, 0.1, mlp_mode=2)
        
        if self.mode=='pre':
            self.proj = nn.Linear(embed_size*(mlp_len),pred_len) #####
            self.flatten = nn.Flatten(start_dim=-2)
            self.trans = MLPBlock(1+t_size , time_embed_size, 1, 0.1, mlp_mode=2)
    def forward(self, x_clip_patch, y_mark):  #(B,T,d_model ) 
        x_clip_patch_att = self.temp_list(x_clip_patch.transpose(1,2)).transpose(1,2) 
        x_clip_patch_att = self.ff(x_clip_patch_att)+x_clip_patch_att
        x = x_clip_patch_att
        
        if self.mode=='pre':
            x = self.proj(self.flatten(x)).reshape(-1,self.pred_len,1)
            x = self.trans(torch.cat([x,y_mark],-1))
        return x   
    
 #趋势项处理部分（论文 §4.3 Trend Information Extraction Module，堆叠 M_t 层）
class TrendInformationExtraction(nn.Module):
    def __init__(self, layer_num, seq_len, pred_len, embed_size, patch_len_list, stride_list ,time_embed_size, t_size,layer_num_cnn):
        super().__init__()    
        self.layer_num = layer_num   
        Fact_list = []
        if layer_num>1:
            for i in range(layer_num-1):
                Fact_list.append(PatchMixing(t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list, layer_num_cnn,mode='trans'))
        Fact_list.append(PatchMixing(t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list, layer_num_cnn, mode='pre'))
        self.Fact_list = nn.ModuleList(Fact_list)
  
    def forward(self, x, y_mark):  # (B,T,d_model)
        for i in range(self.layer_num):    
            x = self.Fact_list[i](x, y_mark)
         
        return x  
######################################################################################################  
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, d_inp):
        super(TimeFeatureEmbedding, self).__init__()
        self.d_inp = d_inp
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)
    
######################################################################################################  
class STMixerv5(nn.Module):

    def __init__(self, args ):
        super(STMixerv5, self).__init__()
        
        self.args = args  
        self.decomp_len = args.decomp_len
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len  
        self.t_size = args.t_size
        self.time_embed_size = args.time_embed_size
        self.coeff = self.args.coeff
        self.decomposition = MultiScaleSeriesDecomposition(args.moving_avg,args.decomp_len,args.seq_len)
        if self.args.elayer_num>=1:
            self.seasonal_mdf = SeasonalMultiDownsamplingFusion(args.sample_fusion, args.coeff, args.elayer_num, args.time_embed_size, args.output_size, self.seq_len, args.pred_len, args.sampling_list, args.dropout, self.time_embed_size, self.t_size, args.singe_sample_dim)
            self.seasonal_covariate_fusion = MLPBlock(len(args.sampling_list)*args.singe_sample_dim+self.t_size,self.args.sample_embed_size,1,args.dropout,2)
            self.skip_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            #self.sea_norm = nn.LayerNorm([self.pred_len,1])
        else:
            self.seasonal_mdf = nn.Linear(self.seq_len, self.pred_len)
        
        if self.args.dlayer_num>=1:
            self.trend_tie = TrendInformationExtraction(args.dlayer_num, self.seq_len, args.pred_len, args.embed_size, args.patch_len_list, args.stride_list ,self.time_embed_size, self.t_size, args.layer_num_cnn)
            #self.conv_patch_list = nn.Conv1d(in_channels=1+self.t_size, out_channels= args.time_embed_size, kernel_size= args.patch_len_list, stride =  args.stride_list)
            #self.conv_patch_list = nn.Conv1d(in_channels=1, out_channels= args.embed_size, kernel_size= args.patch_len_list, stride =  args.stride_list)
            self.conv_patch_list = SingleScalePatching(1, args.embed_size, args.patch_len_list, args.stride_list, args.layer_num_cnn)
            self.skip_Trend = nn.Linear(self.seq_len, self.pred_len)
            #self.tre_norm = nn.LayerNorm([self.pred_len,1])
        else:
            self.trend_tie = nn.Linear(self.seq_len, self.pred_len)
            
        #时间特征嵌入与协变量融合（论文 §4.4 Covariate Fusion Module）
        self.x_mark_embed = TimeFeatureEmbedding(self.t_size, args.mark_size)
        self.y_mark_embed = TimeFeatureEmbedding(self.t_size, args.mark_size)
        self.covariate_fusion = MLPBlock(1+self.t_size,args.time_embed_size,1,args.dropout,2)
    def encoder(self, x, x_mark, y_mark): 
        if self.args.norm==True:
            means = x[:,-self.seq_len:].mean(1, keepdim=True).detach() #B, T, N
            x = x - means #
            stdev = torch.sqrt(torch.var(x[:,-self.seq_len:], dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev #标准化
        y_mark = y_mark[:,-self.args.pred_len:,:]
       
        #(B,T+P,C)
        B,T,C = x.shape
        x = x.transpose(1,2).reshape(B*C,T,1)        
        #(B,T+P,M)
        B,P,M = y_mark.shape        
        y_mark = y_mark.unsqueeze(1).repeat(1,C,1,1).reshape(B*C,P,M)           
        #(B,T+P,M)
        B,T,M = x_mark.shape        
        x_mark = x_mark.unsqueeze(1).repeat(1,C,1,1).reshape(B*C,T,M)[:,-self.seq_len:]  
        
        y_mark = self.y_mark_embed(y_mark)
        x_mark = self.x_mark_embed(x_mark) #（B，V，4）
       
        seasonal_init, trend_init, end_paddings_list = self.decomposition(x)   #（B,T,V）
  
        #对季节项进行下采样处理
        if self.args.elayer_num>=1:
            seasonal_output = self.seasonal_mdf(seasonal_init, x_mark)  #x_mark  
            seasonal_output = self.seasonal_covariate_fusion(torch.cat([torch.cat(seasonal_output,-1),y_mark],-1)) + self.skip_Seasonal(seasonal_init.transpose(1,2)).transpose(1,2) 
            
        else:
            seasonal_output = self.seasonal_mdf(seasonal_init.permute( 0, 2, 1)).permute( 0, 2, 1)
            
        #对趋势项进行局部处理
        if self.args.dlayer_num>=1:  
            trend_init_patch = self.conv_patch_list( trend_init) #(B,PATCH,d_model) CNN
            
            trend_output =  self.trend_tie(trend_init_patch, y_mark) + self.skip_Trend(trend_init.transpose(1,2)).transpose(1,2) #mlp*2
        else:
            trend_output = self.trend_tie(trend_init.transpose(1,2)).transpose(1,2)
        #融合后进行时间信息加强（论文 §4.4 Covariate Fusion Module）
        x = seasonal_output + trend_output  
        x = self.covariate_fusion(torch.cat([x,y_mark],-1) )
        
        
        x = x.reshape(B,C,self.args.pred_len).transpose(1,2) 
        if self.args.norm==True:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1,x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1)) 
            for i in range(len(end_paddings_list)):
                end_paddings_list[i] = end_paddings_list[i].reshape(B,C,-1).transpose(1,2)  * (stdev[:, 0, :].unsqueeze(1).repeat(1,end_paddings_list[i].shape[1], 1))
                end_paddings_list[i] = end_paddings_list[i].reshape(B,C,-1).transpose(1,2)  + (means[:, 0, :].unsqueeze(1).repeat(1, end_paddings_list[i].shape[1], 1)) 

        return x,end_paddings_list  #（B,P,V）

 
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec ): 
        
        dec_out, end_paddings_list = self.encoder(x_enc, x_mark_enc, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], end_paddings_list  # [B, L, D]
 
