# =============================================================================
# ST-Mixer 论文 Table 9 / Figure 12 趋势分支替换变体：ST-Mixer-Top-Down (T)
# （仅替换趋势分支：v5 的趋势局部分支（Encoder_T + Conv_patch + skip_Trend）
#   → 内嵌 TimeMixer（其 PastDecomposableMixing 只做 Top-Down 多尺度趋势
#   混合，方向 = coarse-to-fine（粗→细）：MultiScaleTrendMixing 以
#   up_sampling_layers 自 coarse 端起步、low->high 逐层上采样聚合，与论文
#   §7.2 开头定义一致）；季节性 MDF（Encoder_S + season_trans + skip_Seasonal）、
#   多尺度分解+线性 padding、协变量融合均与 Model/STMixerv5.py 一致；
#   保持通道独立与参数共享）
# 与 Model/STMixerv5.py 的差异清单（除下列条目外逐行一致；助手类名保持不变）：
#   1. 主类名 STMixerv5 → STMixer_Trend_TD（及 super() 调用）。
#   2. 趋势分支替换：v5 的 Linear_Trend(Encoder_T) + conv_patch_list +
#      skip_Trend → Linear_Trend=内嵌 TimeMixer（输出 (B, pred_len, 1)），
#      前向为 Linear_Trend(trend_init, x_mark, None, None)。
#      TimeMixer 专属超参经 getattr 取默认（d_model=128/d_ff=256/e_layers=2 与
#      part1 run.py 默认一致；down_sampling_layers=3/window=2/method='avg'——
#      part1 trend-mixer 脚本（part1 仓库 scripts/STMixer/*/our_modelv5_T_mixer.sh）
#      固定为 2/2/avg，本文件 getattr 回退默认值为 3，本仓库随附的 sh 脚本固定
#      2/2/avg，同 seasonal_bottomup.py，仅影响结构不影响算法）。part1 源文件直接传
#      TimeMixer(args)，本仓库 run.py 无这些属性，故同 Task 5 改 getattr。
#   3. part1 源文件（Model/STMixerv5_trend_mixer.py）的 PastDecomposableMixing
#      实际调用的就是 MultiScaleTrendMixing（Top-Down 方向，up_sampling_layers，
#      coarse-to-fine，与论文定义吻合），但变量/注释沿用了误导性的
#      season_list/"bottom-up season mixing" 命名；此处按论文趋势分支定义做
#      最小改名修正（season_list→trend_list、注释改为 top-down trend mixing），
#      数学逻辑不变。
#   4. 新增内嵌基线类 MultiScaleSeasonMixing/MultiScaleTrendMixing/
#      PastDecomposableMixing/TimeMixer（携带自 part1，未改动）。
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from types import SimpleNamespace
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize


# 下采样之后不再将隐藏层压缩至1
# 在STMIXER2的基础上。对下采样模块的子序列共享MLP参数
# 在STMIXER3的基础上，将end_padding部分加入损失函数
class flatten(nn.Module):
    def forward(self, input):
        return torch.reshape(input, (input.size(0), -1))


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        x = x[:, -336:, :]
        x = torch.transpose(x, 1, 2)
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[:, :, 0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq < torch.min(top_k_freq, dim=-1, keepdim=True)[0]] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        x_season = torch.transpose(x_season, 1, 2)
        x_trend = torch.transpose(x_trend, 1, 2)
        self.x_season = x_season
        self.x_trend = x_trend
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = 1

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        # （part1 源文件此处未做真实分解，直接对各尺度嵌入序列做 Top-Down 趋势
        #   混合；保持原逻辑不变，仅修正误导性命名，见文件头差异清单第 3 条）
        trend_list = []
        for x in x_list:
            trend_list.append(x.permute(0, 2, 1))

        # top-down trend mixing (coarse-to-fine)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # # bottom-up season mixing（part1 中本已注释，保持未启用）
        # out_season_list = self.mixing_multi_scale_season(season_list)

        out_list = []
        for ori, out_trend, length in zip(x_list, out_trend_list,
                                          length_list):
            out = out_trend
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = 1
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])


        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, None,
                                                  configs.dropout)


        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(None, affine=False, non_norm= False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.projection_layer = nn.Linear(
            configs.d_model, 1, bias=True)

        self.out_res_layers = torch.nn.ModuleList([
            torch.nn.Linear(
                configs.seq_len // (configs.down_sampling_window ** i),
                configs.seq_len // (configs.down_sampling_window ** i),
            )
            for i in range(configs.down_sampling_layers + 1)
        ])

        self.regression_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )


    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        return (x_list, None)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            # print(x_enc_ori.shape)
            x_enc_sampling = down_pool(x_enc_ori)
            # print(x_enc_sampling.shape)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc


    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, -1, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print(x_enc.shape, x_mark_enc.shape)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc) # 池化/间隔采样

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                # print(x.shape, x_mark.shape)
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
            # print(x.shape, x_mark.shape)
            enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
            enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # print(x_enc.shape, x_dec.shape, x_mark_enc.shape, x_mark_dec.shape)
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out

    ############################################################################################################


# 时间序列分解
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride, decomp_len, seq_len):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        # self.endd = nn.Linear(self.decomp_len,math.floor((self.kernel_size - 1) // 2))
        self.endd = MLPBlock(self.decomp_len, self.decomp_len // 2, math.floor((self.kernel_size - 1) // 2), mlp_mode=2)

    def forward(self, x):
        end = self.endd(x.transpose(1, 2)).transpose(1, 2)
        self.end = end
        x = torch.cat([x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x, end


# 单尺度分解
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size, decomp_len, seq_len):
        super(series_decomp, self).__init__()
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.moving_avg = moving_avg(kernel_size, 1, decomp_len, seq_len)

    def forward(self, x):
        moving_mean, end_padding = self.moving_avg(x)
        res = x[:, -moving_mean.shape[1]:, :] - moving_mean
        return res[:, -self.seq_len:, :], moving_mean[:, -self.seq_len:, :], end_padding


# 多尺度分解
class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block
    """

    def __init__(self, kernel_size, decomp_len, seq_len):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.decomp_len = decomp_len
        self.seq_len = seq_len
        self.series_decomp = nn.ModuleList([series_decomp(kernel, decomp_len, seq_len) for kernel in kernel_size])

    def forward(self, x):
        x = x
        moving_mean = []
        res = []
        end_paddings = []
        for func in self.series_decomp:
            sea, moving_avg, end_padding = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)
            end_paddings.append(end_padding)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        self.sea = sea
        self.moving_mean = moving_mean
        return sea, moving_mean, end_paddings


############################################################################################################
# 对季节项下采样

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, dropout=0.1, mlp_mode=1):
        super().__init__()
        self.mlp_mode = mlp_mode  # MLP多层还是单层
        if mlp_mode == 2:
            self.fc1 = nn.Linear(input_dim, mlp_dim)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(mlp_dim, output_dim)
        elif mlp_mode == 1:
            self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        if self.mlp_mode == 2:
            return self.fc2(self.dropout(self.gelu(self.fc1(x))))
        elif self.mlp_mode == 1:
            return self.fc1(x)


# 单一粒度下采样
class FactorizedTemporalMixing(nn.Module):
    def __init__(self, sample_fusion, sampling_list, current_layer, coeff, seq_len, pred_len, sampling, dropout,
                 singe_sample_dim, mode='pre'):
        super().__init__()
        self.mode = mode

        assert sampling in [1, 2, 3, 4, 6, 8, 12, 24, 48]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sampling = sampling
        self.coeff = coeff
        self.current_layer = current_layer
        self.singe_sample_dim = singe_sample_dim
        if current_layer == 0:  # 第一层和中间层的输入维度有区别，因此需要作出区分
            # self.temporal_fac = nn.ModuleList([
            #     MLPBlock(seq_len // sampling, int(coeff * seq_len*(1+singe_sample_dim) // sampling // 2) , singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2) for _ in range(sampling)
            # ])
            self.temporal_fac = MLPBlock(seq_len // sampling,
                                         int(coeff * seq_len * (1 + singe_sample_dim) // sampling // 2),
                                         singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2)
        else:
            # self.temporal_fac = nn.ModuleList([
            #     MLPBlock(sample_fusion * seq_len // sampling,int( coeff * (sample_fusion+singe_sample_dim) * seq_len // sampling // 2) , singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2) for _ in range(sampling)
            # ])
            self.temporal_fac = MLPBlock(sample_fusion * seq_len // sampling,
                                         int(coeff * (sample_fusion + singe_sample_dim) * seq_len // sampling // 2),
                                         singe_sample_dim * seq_len // sampling, dropout, mlp_mode=2)

        if self.mode == 'pre':  # 当位于最后一层时，需要额外映射为未来值
            self.predict = MLPBlock(seq_len, int(coeff * (seq_len * singe_sample_dim + pred_len) // 2), pred_len,
                                    dropout, mlp_mode=2)

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        if self.mode == 'pre':
            x = x.transpose(1, 2)  # (B,C,T)
            x_samp = []
            # for i in range(len(self.temporal_fac)):
            # x_samp.append(  torch.reshape(self.temporal_fac[i](flatten()(x[:, :, i::self.sampling])), (-1,self.singe_sample_dim,self.seq_len // self.sampling))  )
            #  #(B,C,T)

            x_to_trans = torch.stack([x[:, :, i::self.sampling] for i in range(self.sampling)], dim=1).view(x.shape[0],
                                                                                                            self.sampling,
                                                                                                            -1)  # (B,S,C*T/S)
            x = torch.reshape(torch.reshape(self.temporal_fac(x_to_trans),
                                            (x.shape[0], self.sampling, self.singe_sample_dim,
                                             self.seq_len // self.sampling)).permute(0, 2, 3, 1),
                              (x.shape[0], self.singe_sample_dim, -1)).transpose(1, 2)  # (B,T,C)

            # x = self.merge((x.shape[0],self.singe_sample_dim,self.seq_len), x_samp).transpose(1,2)  #(B,C,T) ==>#(B,T,C)
            x = self.predict(x.transpose(1, 2)).transpose(1, 2)
            return x  # (B,T,C)

        elif self.mode == 'trans':
            x = x.transpose(1, 2)  # (B,C,T)
            x_samp = []
            # for i in range(len(self.temporal_fac)):
            # x_samp.append(  torch.reshape(self.temporal_fac[i]( flatten()(x[:, :, i::self.sampling]) ),((-1,self.singe_sample_dim,self.seq_len // self.sampling))) ) #(B,C,T)

            x_to_trans = torch.stack([x[:, :, i::self.sampling] for i in range(self.sampling)], dim=1).view(x.shape[0],
                                                                                                            self.sampling,
                                                                                                            -1)  # (B,S,C*T/S)
            x = torch.reshape(torch.reshape(self.temporal_fac(x_to_trans), (
                x.shape[0], self.sampling, self.singe_sample_dim, self.seq_len // self.sampling)).permute(0, 2, 3, 1),
                              (x.shape[0], self.singe_sample_dim, -1)).transpose(1, 2)  # (B,T,C)

            # x = self.merge((x.shape[0],self.singe_sample_dim,self.seq_len), x_samp).transpose(1,2)  #(B,C,T) ==>#(B,T,C)
            return x  # (B,T,C)


# 下采样融合模块
class EncoderLayer(nn.Module):
    def __init__(self, sample_fusion, coeff, seq_len, pred_len,
                 sampling_list, dropout, mode, singe_sample_dim, current_layer=None):
        super().__init__()
        self.sampling_list = sampling_list
        self.mode = mode
        self.coeff = coeff
        Fact_list = []
        for i in range(len(sampling_list)):
            Fact_list.append(
                FactorizedTemporalMixing(sample_fusion, sampling_list, current_layer, coeff, seq_len, pred_len,
                                         sampling_list[i], dropout, singe_sample_dim, mode))
        self.Fact_list = nn.ModuleList(Fact_list)

        if mode == 'trans':
            self.merge = MLPBlock(len(sampling_list) * singe_sample_dim,
                                  int(self.coeff * ((len(sampling_list) * singe_sample_dim + sample_fusion) // 2)),
                                  sample_fusion, dropout, mlp_mode=2)

    def forward(self, x, x_mark):  # (B,T,d_model)
        x_pre = []
        for i in range(len(self.sampling_list)):  # 根据不同的下采样细粒度进行对数据的重新处理
            x_pre.append(self.Fact_list[i](x))

        if self.mode == 'trans':
            x_pre = self.merge(torch.cat(x_pre, -1))  # 对不同下采样细粒度的处理结果进行合并
        return x_pre


# 季节项处理部分
class Encoder_S(nn.Module):
    def __init__(self, sample_fusion, coeff, layer_num, embed_size, output_size, seq_len, pred_len, sampling_list,
                 dropout, time_embed_size, t_size, singe_sample_dim):
        super().__init__()
        self.sampling_list = sampling_list
        self.layer_num = layer_num
        Fact_list = []
        if layer_num > 1:
            for i in range(layer_num - 1):
                Fact_list.append(
                    EncoderLayer(sample_fusion, coeff,  seq_len,
                                 pred_len, sampling_list, dropout, 'trans', singe_sample_dim,
                                 current_layer=i))  # trans:中间层不需要进行预测,,current_layer从0开始计数，等于0时，将影响到FactorizedTemporalMixing的输入的维度。
            Fact_list.append(
                EncoderLayer(sample_fusion, coeff, seq_len, pred_len,
                             sampling_list, dropout, 'pre', singe_sample_dim))  # pre:最后一层不需要进行预测

        elif layer_num == 1:
            Fact_list.append(
                EncoderLayer(sample_fusion, coeff, seq_len, pred_len,
                             sampling_list, dropout, 'pre', singe_sample_dim, current_layer=0))

        self.Fact_list = nn.ModuleList(Fact_list)

    def forward(self, x, x_mark):  # (B,T,d_model)
        for i in range(self.layer_num):
            x = self.Fact_list[i](x, x_mark)

        return x

    ###############################################################################################################


# 进行局部信息的提取
class Conv_patch(nn.Module):
    def __init__(self, in_channels, out_channels, patch_len, stride, layer_num_cnn):
        super(Conv_patch, self).__init__()

        self.conv_patch = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_len, stride=stride),
            *[nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=patch_len, stride=stride)] * (
                        layer_num_cnn - 1))
        # self.conv_patch = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=patch_len, stride = stride)

    def forward(self, x):
        x_patch = self.conv_patch(x.transpose(1, 2)).transpose(1, 2)
        return x_patch


# 通过两组MLP进行局部信息内部和外部的交互
class patch_attention(nn.Module):
    def __init__(self, t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list,
                 layer_num_cnn, mode='pre'):
        super(patch_attention, self).__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.pred_len = pred_len
        self.mode = mode
        mlp_len = seq_len
        for i in range(layer_num_cnn):
            mlp_len = (mlp_len - patch_len_list) // stride_list + 1
        self.temp_list = nn.Sequential(nn.Linear(mlp_len, mlp_len), nn.GELU(), nn.Linear(mlp_len, mlp_len))

        self.ff = MLPBlock(embed_size, 2 * embed_size, embed_size, 0.1, mlp_mode=2)

        if self.mode == 'pre':
            self.proj = nn.Linear(embed_size * (mlp_len), pred_len)  #####
            self.flatten = nn.Flatten(start_dim=-2)
            self.trans = MLPBlock(1 + t_size, time_embed_size, 1, 0.1, mlp_mode=2)

    def forward(self, x_clip_patch, y_mark):  # (B,T,d_model )
        x_clip_patch_att = self.temp_list(x_clip_patch.transpose(1, 2)).transpose(1, 2)
        x_clip_patch_att = self.ff(x_clip_patch_att) + x_clip_patch_att
        x = x_clip_patch_att

        if self.mode == 'pre':
            x = self.proj(self.flatten(x)).reshape(-1, self.pred_len, 1)
            x = self.trans(torch.cat([x, y_mark], -1))
        return x

        # 趋势项处理部分


class Encoder_T(nn.Module):
    def __init__(self, layer_num, seq_len, pred_len, embed_size, patch_len_list, stride_list, time_embed_size, t_size,
                 layer_num_cnn):
        super().__init__()
        self.layer_num = layer_num
        Fact_list = []
        if layer_num > 1:
            for i in range(layer_num - 1):
                Fact_list.append(
                    patch_attention(t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list,
                                    layer_num_cnn, mode='trans'))
        Fact_list.append(
            patch_attention(t_size, time_embed_size, seq_len, pred_len, embed_size, patch_len_list, stride_list,
                            layer_num_cnn, mode='pre'))
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
class STMixer_Trend_TD(nn.Module):

    def __init__(self, args):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(STMixer_Trend_TD, self).__init__()

        self.args = args
        self.decomp_len = args.decomp_len
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.t_size = args.t_size
        self.time_embed_size = args.time_embed_size
        self.coeff = self.args.coeff
        self.decompsition = series_decomp_multi(args.moving_avg, args.decomp_len, args.seq_len)
        # self.decompsition = DFT_series_decomp(top_k=3)
        if self.args.elayer_num >= 1:
            self.Linear_Seasonal = Encoder_S(args.sample_fusion, args.coeff, args.elayer_num, args.time_embed_size,
                                             args.output_size, self.seq_len, args.pred_len, args.sampling_list,
                                             args.dropout, self.time_embed_size, self.t_size, args.singe_sample_dim)
            self.season_trans = MLPBlock(len(args.sampling_list) * args.singe_sample_dim + self.t_size,
                                         self.args.sample_embed_size, 1, args.dropout, 2)
            self.skip_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            # self.sea_norm = nn.LayerNorm([self.pred_len,1])
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)

        if self.args.dlayer_num >= 1:
            # TimeMixer 专属超参：目标仓库 run.py 未定义，经 getattr 取默认
            #（同 seasonal_bottomup.py：d_model/d_ff/e_layers 与 part1 run.py 默认
            #  一致；down_sampling 取 TimeMixer 官方 ETT 配置 layers=3/window=2/avg）
            tm_configs = SimpleNamespace(
                seq_len=self.seq_len,
                label_len=getattr(args, 'label_len', 48),
                pred_len=self.pred_len,
                d_model=getattr(args, 'd_model', 128),
                d_ff=getattr(args, 'd_ff', 256),
                e_layers=getattr(args, 'e_layers', 2),
                dropout=args.dropout,
                down_sampling_window=getattr(args, 'down_sampling_window', 2),
                down_sampling_layers=getattr(args, 'down_sampling_layers', 3),
                down_sampling_method=getattr(args, 'down_sampling_method', 'avg'),
            )
            self.Linear_Trend = TimeMixer(tm_configs)
        else:
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.markx = TimeFeatureEmbedding(self.t_size, args.mark_size)
        self.marky = TimeFeatureEmbedding(self.t_size, args.mark_size)
        self.x_trans = MLPBlock(1 + self.t_size, args.time_embed_size, 1, args.dropout, 2)

    def encoder(self, x, x_mark, y_mark):
        if self.args.norm == True:
            means = x[:, -self.seq_len:].mean(1, keepdim=True).detach()  # B, T, N
            x = x - means  #
            stdev = torch.sqrt(torch.var(x[:, -self.seq_len:], dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev  # 标准化
        y_mark = y_mark[:, -self.args.pred_len:, :]

        # (B,T+P,C)
        B, T, C = x.shape
        x = x.transpose(1, 2).reshape(B * C, T, 1)
        # (B,T+P,M)
        B, P, M = y_mark.shape
        y_mark = y_mark.unsqueeze(1).repeat(1, C, 1, 1).reshape(B * C, P, M)
        # (B,T+P,M)
        B, T, M = x_mark.shape
        x_mark = x_mark.unsqueeze(1).repeat(1, C, 1, 1).reshape(B * C, T, M)[:, -self.seq_len:]

        y_mark = self.marky(y_mark)
        x_mark = self.markx(x_mark)  # （B，V，4）

        # seasonal_init, trend_init = self.decompsition(x.to(torch.device('cuda:{}'.format(self.args.gpu))))   #（B,T,V）
        seasonal_init, trend_init, end_paddings_list = self.decompsition(x)  # （B,T,V）
        # print(seasonal_init.shape, trend_init.shape)
        # 对季节项进行下采样处理
        if self.args.elayer_num >= 1:
            seasonal_output = self.Linear_Seasonal(seasonal_init, x_mark)  # x_mark
            seasonal_output = self.season_trans(
                torch.cat([torch.cat(seasonal_output, -1), y_mark], -1)) + self.skip_Seasonal(
                seasonal_init.transpose(1, 2)).transpose(1, 2)

        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 对趋势项进行局部处理
        if self.args.dlayer_num >= 1:
            # print(seasonal_init.shape, trend_init.shape, x_mark.shape)
            trend_output = self.Linear_Trend(trend_init, x_mark, None, None) 
        else:
            trend_output = self.Linear_Trend(trend_init.transpose(1, 2) ).transpose(1, 2)
        # 融合后进行时间信息加强
        x = seasonal_output + trend_output
        x = self.x_trans(torch.cat([x, y_mark], -1))

        x = x.reshape(B, C, self.args.pred_len).transpose(1, 2)
        if self.args.norm == True:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            for i in range(len(end_paddings_list)):
                end_paddings_list[i] = end_paddings_list[i].reshape(B, C, -1).transpose(1, 2) * (
                    stdev[:, 0, :].unsqueeze(1).repeat(1, end_paddings_list[i].shape[1], 1))
                end_paddings_list[i] = end_paddings_list[i].reshape(B, C, -1).transpose(1, 2) + (
                    means[:, 0, :].unsqueeze(1).repeat(1, end_paddings_list[i].shape[1], 1))

        return x, end_paddings_list  # （B,P,V）

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        dec_out, end_paddings_list = self.encoder(x_enc, x_mark_enc, x_mark_dec)
        return dec_out[:, -self.pred_len:, :], end_paddings_list  # [B, L, D]

