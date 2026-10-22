# exp 骨架：模型注册（MODEL_REGISTRY）与训练/验证/测试主流程（Exp_Main）
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from exp.Main_basic import Exp_Basic
from Model.STMixerv5 import STMixerv5
from Model.ablation.ablation_tie import STMixer_Ablation_TIE
from Model.ablation.ablation_scale import STMixer_Ablation_Scale
from Model.ablation.ablation_padding import STMixer_Ablation_Padding
from Model.ablation.ablation_covariate import STMixer_Ablation_Covariate
from Model.component_swap.tie_mdf import STMixer_Swap_TIEMDF
from Model.component_swap.mdf_mdf import STMixer_Swap_MDFMDF
from Model.component_swap.tie_tie import STMixer_Swap_TIETIE
from Model.component_swap.mdf_only import STMixer_Swap_MDF
from Model.component_swap.tie_only import STMixer_Swap_TIE
from Model.seasonal_variants.seasonal_sci import STMixer_Seasonal_SCI
from Model.seasonal_variants.seasonal_mts import STMixer_Seasonal_MTS
from Model.seasonal_variants.seasonal_bottomup import STMixer_Seasonal_BU
from Model.seasonal_variants.seasonal_topdown import STMixer_Seasonal_TD
from Model.trend_variants.trend_mean import STMixer_Trend_Mean
from Model.trend_variants.trend_linear import STMixer_Trend_Linear
from Model.trend_variants.trend_topdown import STMixer_Trend_TD
# Table 4/5 自跑 baseline（Task 7，迁移自 part1 Model/）
from Model.baselines.itransformer import iTransformer
from Model.baselines.patchtst import patchtst as PatchTST
from Model.baselines.timemixer import TimeMixer
from Model.baselines.semixer import SEMixer
from Model.baselines.mixlinear import MixLinear
from Model.baselines.dlinear import Dlinear
from Model.baselines.linear import Linear
from Model.baselines.nlinear import Nlinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import DataLoader
from Data.Data_load import *
from dateutil.relativedelta import relativedelta
import datetime
from fvcore.nn import FlopCountAnalysis
warnings.filterwarnings('ignore')

MODEL_REGISTRY = {
    'STMixerv5': STMixerv5,
    # §5.6 / Figure 8 消融变体（Task 3）
    'STMixer_Ablation_TIE': STMixer_Ablation_TIE,
    'STMixer_Ablation_Scale': STMixer_Ablation_Scale,
    'STMixer_Ablation_Padding': STMixer_Ablation_Padding,
    'STMixer_Ablation_Covariate': STMixer_Ablation_Covariate,
    # Table 7 / Figure 10 组件交换变体（Task 4）
    'STMixer_Swap_TIEMDF': STMixer_Swap_TIEMDF,
    'STMixer_Swap_MDFMDF': STMixer_Swap_MDFMDF,
    'STMixer_Swap_TIETIE': STMixer_Swap_TIETIE,
    'STMixer_Swap_MDF': STMixer_Swap_MDF,
    'STMixer_Swap_TIE': STMixer_Swap_TIE,
    # Table 8 / Figure 11 季节性分支替换变体（Task 5）
    'STMixer_Seasonal_SCI': STMixer_Seasonal_SCI,
    'STMixer_Seasonal_MTS': STMixer_Seasonal_MTS,
    'STMixer_Seasonal_BU': STMixer_Seasonal_BU,
    'STMixer_Seasonal_TD': STMixer_Seasonal_TD,
    # Table 9 / Figure 12 趋势分支替换变体（Task 6）
    'STMixer_Trend_Mean': STMixer_Trend_Mean,
    'STMixer_Trend_Linear': STMixer_Trend_Linear,
    'STMixer_Trend_TD': STMixer_Trend_TD,
    # Table 4/5 自跑 baseline（Task 7）
    'iTransformer': iTransformer,
    'PatchTST': PatchTST,
    'TimeMixer': TimeMixer,
    'SEMixer': SEMixer,
    'MixLinear': MixLinear,
    'DLinear': Dlinear,
    'Linear': Linear,
    'NLinear': Nlinear,
}

STMIXER_V5_MODELS = [
    'STMixerv5',
    # §5.6 / Figure 8 消融变体：均为 v5 架构，走 (outputs, end_paddings_list) 分支
    'STMixer_Ablation_TIE',
    'STMixer_Ablation_Scale',
    'STMixer_Ablation_Padding',
    'STMixer_Ablation_Covariate',
    # Table 7 / Figure 10 组件交换变体：均为 v5 架构，走 (outputs, end_paddings_list) 分支
    'STMixer_Swap_TIEMDF',
    'STMixer_Swap_MDFMDF',
    'STMixer_Swap_TIETIE',
    'STMixer_Swap_MDF',
    'STMixer_Swap_TIE',
    # Table 8 / Figure 11 季节性分支替换变体：均为 v5 架构，走 (outputs, end_paddings_list) 分支
    'STMixer_Seasonal_SCI',
    'STMixer_Seasonal_MTS',
    'STMixer_Seasonal_BU',
    'STMixer_Seasonal_TD',
    # Table 9 / Figure 12 趋势分支替换变体：均为 v5 架构，走 (outputs, end_paddings_list) 分支
    'STMixer_Trend_Mean',
    'STMixer_Trend_Linear',
    'STMixer_Trend_TD',
]


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        print("building model:", self.args.model)
        model = MODEL_REGISTRY[self.args.model](self.args)
        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameters: %.2fM' % (total / 1e6))
        return model

    def _get_data(self, flag):
        print("loading data")
        _, data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        print("selecting optimizer")
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        print("selecting criterion")
        criterion = nn.MSELoss()
        return criterion

    def _compute_base_loss(self, outputs, batch_y, criterion):
        if self.args.loss == 'all':
            return criterion[0](outputs, batch_y) + criterion[1](outputs, batch_y)
        return criterion(outputs, batch_y)

    def _compute_end_padding_loss(self, end_paddings_list, batch_y, criterion):
        if not end_paddings_list:
            return 0
        end_padding_loss = 0
        for ep in end_paddings_list:
            target = batch_y[:, :ep.shape[1], :]
            if self.args.loss == 'all':
                end_padding_loss += criterion[0](ep, target) + criterion[1](ep, target)
            else:
                end_padding_loss += criterion(ep, target)
        return end_padding_loss

    def _move_batch_to_device(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_x_mark.float().to(self.device),
            batch_y_mark.float().to(self.device),
        )

    def _model_forward(self, batch_x, batch_x_mark, batch_y, batch_y_mark):
        if self.args.model in STMIXER_V5_MODELS:
            outputs, end_paddings_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            return outputs, end_paddings_list, None
        elif self.args.model in ['ATAformer']:
            outputs, loss_constrast = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            return outputs, None, loss_constrast
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            return outputs, None, None

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        min_val_loss = []

        fff_time = time.time()
        try:
            import io
            import sys
            from contextlib import redirect_stderr
            seq_len = self.args.decomp_len if 'STMixer' in str(self.model) else self.args.seq_len
            input_tensor1 = torch.randn(self.args.batch_size, seq_len, self.args.input_size).to(self.device)
            input_tensor2 = torch.randn(self.args.batch_size, seq_len, self.args.mark_size).to(self.device)
            input_tensor3 = torch.randn(self.args.batch_size, self.args.pred_len, self.args.input_size).to(self.device)
            input_tensor4 = torch.randn(self.args.batch_size, self.args.pred_len + self.args.label_len, self.args.mark_size).to(self.device)

            max_memory_used_test = 0
            torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])
            # 完全禁用 fvcore 警告输出
            with redirect_stderr(io.StringIO()):
                for _ in range(100):
                    flops = FlopCountAnalysis(self.model, (input_tensor1, input_tensor2, input_tensor3, input_tensor4))
                    current_max_memory_test = torch.cuda.max_memory_allocated(device=self.args.gpu[0])
                    if current_max_memory_test > max_memory_used_test:
                        max_memory_used_test = current_max_memory_test
            cost_time = (time.time() - fff_time) / 100
            print(f"Total Train FLOPs: {flops.total()}", "FPS: {}".format(1 / cost_time))
            print(f"Maximum GPU memory allocated during training: {max_memory_used_test / (1024**2):.2f} MB")
        except:
            pass

        max_memory_used = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_v = []
            self.model.train()
            epoch_time = time.time()
            torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs, end_paddings_list, loss_constrast = self._model_forward(
                    batch_x, batch_x_mark, batch_y, batch_y_mark)

                base_loss = self._compute_base_loss(outputs, batch_y, criterion)
                reg_loss = regularization(self.model, self.args.weight_decay, self.args.p) if self.args.weight_decay > 0 else 0

                extra_loss = 0
                if end_paddings_list is not None:
                    ep_loss = self._compute_end_padding_loss(end_paddings_list, batch_y, criterion)
                    ep_scale = 1.0 if self.args.weight_decay > 0 else max(1, len(end_paddings_list))
                    extra_loss = ep_loss / ep_scale
                elif loss_constrast is not None:
                    extra_loss = loss_constrast

                loss = base_loss + reg_loss + extra_loss

                train_loss.append(loss.item())
                train_loss_v.append(base_loss.item())

                if (i + 1) % self.args.report_num == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                current_max_memory = torch.cuda.max_memory_allocated(device=self.args.gpu[0])
                if current_max_memory > max_memory_used:
                    max_memory_used = current_max_memory

            print(f"Epoch: {epoch + 1} Maximum GPU memory allocated during training: {max_memory_used / (1024**2):.2f} MB")

            total_params = sum(p.numel() for p in self.model.parameters())
            print(f'current {epoch + 1} start time: {datetime.datetime.now() + relativedelta(hours=8)} ')
            print(f'total parameters {total_params} ')
            total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'training parameters {total_trainable_params} ')

            print("Epoch: {} train cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_v = np.average(train_loss_v)
            if epoch % self.args.val_time == 0:
                vali_time = time.time()
                vali_loss = self.vali(vali_loader, criterion)
                print("Epoch: {} vali cost time: {}".format(epoch + 1, time.time() - vali_time))
                test_time = time.time()
                if epoch % 5 == 1000:
                    test_loss = self.vali(test_loader, criterion)
                else:
                    test_loss = vali_loss
                print("Epoch: {} test cost time: {}".format(epoch + 1, time.time() - test_time))

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Loss wo l2: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss, train_loss_v, vali_loss, test_loss))
                if np.isnan(train_loss):
                    break
                min_val_loss.append(vali_loss)
                early_stopping(vali_loss, self.model, path, epoch + 1)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Loss wo l2 end: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, train_loss_v))
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print("train over")
        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.model in STMIXER_V5_MODELS:
                    outputs, end_paddings_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    pred, true = outputs, batch_y
                    ep_loss = self._compute_end_padding_loss(end_paddings_list, true, criterion)
                    loss = criterion(pred, true) + ep_loss / max(1, len(end_paddings_list))
                else:
                    if self.args.model in ['ATAformer']:
                        outputs, _ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss.item())
                del pred
                del true

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        mses = 0
        maes = 0
        count = 0
        folder_path = './test_results_picts/' + self.args.data_set + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print("start test")
        infer_wall_start = time.time()  # 记录整个测试集推理的总墙钟时间
        infer_forward_time = 0.0        # 累计纯模型 forward（排除数据搬运等开销）的时间
        num_samples = 0                 # 测试集总样本数
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                    batch_x, batch_y, batch_x_mark, batch_y_mark)

                t0 = time.time()
                if self.args.model in STMIXER_V5_MODELS:
                    outputs, _ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                infer_forward_time += time.time() - t0
                num_samples += outputs.shape[0]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, -336:, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, -336:, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                count += pred.reshape(-1).shape[0]
                maes += sum(np.abs(pred.reshape(-1) - true.reshape(-1)))
                mses += sum((pred.reshape(-1) - true.reshape(-1)) ** 2)
                del pred
                del true
        maes = maes / count
        mses = mses / count
        print('mse:{}, mae:{}'.format(mses, maes))

        infer_wall_total = time.time() - infer_wall_start
        infer_avg_ms = infer_forward_time / num_samples * 1000 if num_samples > 0 else 0.0
        infer_throughput = num_samples / infer_forward_time if infer_forward_time > 0 else 0.0
        print('Inference time (wall): total {:.4f} s'.format(infer_wall_total))
        print('Inference time (forward): total {:.4f} s, avg {:.4f} ms/sample, throughput {:.2f} samples/s'.format(
            infer_forward_time, infer_avg_ms, infer_throughput))

        return self.model


def regularization(model, weight_decay, p=2):
    reg_loss = sum(
        torch.norm(param, p=p) ** 2
        for name, param in model.state_dict().items()
        if 'weight' in name and 'norm' not in name
    )
    return weight_decay * reg_loss
