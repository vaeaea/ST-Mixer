# LMTS 评测入口：模型注册表与训练/验证/测试主流程（注册表与 exp/Main.py 单一来源）
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from exp.Main_basic import Exp_Basic
from exp.Main import MODEL_REGISTRY, STMIXER_V5_MODELS
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from Data.Data_load_7dataV3 import *
from dateutil.relativedelta import relativedelta
import datetime
from fvcore.nn import FlopCountAnalysis
warnings.filterwarnings('ignore')


def MAE11(pred, true):
    return np.abs(pred - true)


def MSE11(pred, true):
    return (pred - true) ** 2


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
        if self.args.loss == 'mse':
            return nn.MSELoss()
        if self.args.loss == 'mae':
            return nn.L1Loss()
        if self.args.loss == 'all':
            return nn.L1Loss(), nn.MSELoss()

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
            return outputs, end_paddings_list
        else:
            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            return outputs, None

    def _create_flops_inputs(self, batch_size, input_size):
        seq_len = self.args.decomp_len if 'STMixer' in str(self.model) else self.args.seq_len
        return (
            torch.randn(batch_size, seq_len, input_size).to(self.device),
            torch.randn(batch_size, seq_len, self.args.mark_size).to(self.device),
            torch.randn(batch_size, self.args.pred_len, input_size).to(self.device),
            torch.randn(batch_size, self.args.pred_len + self.args.label_len, self.args.mark_size).to(self.device),
        )

    def _measure_flops(self, batch_size, input_size, label="Train"):
        fff_time = time.time()
        inputs = self._create_flops_inputs(batch_size, input_size)
        max_memory_used_test = 0
        torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])
        for _ in range(100):
            flops = FlopCountAnalysis(self.model, inputs)
            current_max_memory_test = torch.cuda.max_memory_allocated(device=self.args.gpu[0])
            if current_max_memory_test > max_memory_used_test:
                max_memory_used_test = current_max_memory_test
        cost_time = (time.time() - fff_time) / 100
        print(f"Total {label} FLOPs: {flops.total()}", "FPS: {}".format(1 / cost_time))
        print(f"Maximum GPU memory allocated during training: {max_memory_used_test / (1024**2):.2f} MB")

    def _get_pred_true(self, outputs, batch_y):
        if self.args.pred_len == 720:
            return outputs.detach().cpu().numpy().astype(np.float16), batch_y.detach().cpu().numpy().astype(np.float16)
        return outputs.detach().cpu().numpy(), batch_y.detach().cpu().numpy()

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

        self._measure_flops(self.args.batch_size, self.args.input_size, "Train")

        max_memory_used = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_v = []
            self.model.train()
            epoch_time = time.time()
            torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])

            for _ in range(self.args.report_freq):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                        batch_x, batch_y, batch_x_mark, batch_y_mark)

                    outputs, end_paddings_list = self._model_forward(
                        batch_x, batch_x_mark, batch_y, batch_y_mark)

                    base_loss = self._compute_base_loss(outputs, batch_y, criterion)
                    reg_loss = regularization(self.model, self.args.weight_decay, self.args.p) if self.args.weight_decay > 0 else 0

                    extra_loss = 0
                    if end_paddings_list is not None:
                        ep_loss = self._compute_end_padding_loss(end_paddings_list, batch_y, criterion)
                        ep_scale = 1.0 if self.args.weight_decay > 0 else len(end_paddings_list)
                        extra_loss = ep_loss / ep_scale

                    loss = base_loss + reg_loss + extra_loss

                    train_loss.append(loss.item())
                    train_loss_v.append(base_loss.item())

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
            print('iter time', iter_count / (time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_v = np.average(train_loss_v)
            if epoch % self.args.val_time == 0:
                vali_time = time.time()
                vali_loss = self.vali(vali_loader, criterion)
                print("Epoch: {} vali cost time: {}".format(epoch + 1, time.time() - vali_time))
                test_time = time.time()
                if epoch == 1000:
                    test_loss = self.vali(test_loader, criterion)
                else:
                    test_loss = vali_loss
                print("Epoch: {} test cost time: {}".format(epoch + 1, time.time() - test_time))

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Loss wo l2 end: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
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
        maes = 0
        mses = 0
        count = 0
        self.model.eval()
        self._measure_flops(256, 1, "vali")

        max_memory_used = 0
        torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])

        with torch.no_grad():
            for ii in range(8 + 7):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader[ii]):
                    batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                        batch_x, batch_y, batch_x_mark, batch_y_mark)

                    outputs, end_paddings_list = self._model_forward(
                        batch_x, batch_x_mark, batch_y, batch_y_mark)
                    pred, true = self._get_pred_true(outputs, batch_y)
                    count += pred.reshape(-1).shape[0]

                    if end_paddings_list is not None:
                        if self.args.loss == 'mae':
                            ep_loss = sum(
                                sum(MAE11(ep.detach().cpu().numpy().reshape(-1), true[:, :ep.shape[1], :].reshape(-1)))
                                for ep in end_paddings_list
                            )
                            maes += sum(MAE11(pred.reshape(-1), true.reshape(-1))) + ep_loss / len(end_paddings_list)
                        elif self.args.loss == 'mse':
                            ep_loss = sum(
                                sum(MSE11(ep.detach().cpu().numpy().reshape(-1), true[:, :ep.shape[1], :].reshape(-1)))
                                for ep in end_paddings_list
                            )
                            mses += sum(MSE11(pred.reshape(-1), true.reshape(-1))) + ep_loss / len(end_paddings_list)
                    else:
                        if self.args.loss == 'mae':
                            maes += sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                        elif self.args.loss == 'mse':
                            mses += sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                    del pred
                    del true

                    current_max_memory = torch.cuda.max_memory_allocated(device=self.args.gpu[0])
                    if current_max_memory > max_memory_used:
                        max_memory_used = current_max_memory

        print(f"Maximum GPU memory allocated during valiing: {max_memory_used / (1024**2):.2f} MB")
        if self.args.loss == 'mae':
            maes = maes / count
            self.model.train()
            return maes
        elif self.args.loss == 'mse':
            mses = mses / count
            self.model.train()
            return mses

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
        result_dirs = ['all_pred_result', 'all_true_result', 'all_mse_result', 'all_mae_result']
        for d in result_dirs:
            full_path = './' + d + '/' + self.args.model + self.args.data_set + '/'
            if not os.path.exists(full_path):
                os.makedirs(full_path)
        f_720_mse = open('./all_mse_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        f_720_mae = open('./all_mae_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        with torch.no_grad():
            for ii in range(8 + 7):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader[ii]):
                    batch_x, batch_y, batch_x_mark, batch_y_mark = self._move_batch_to_device(
                        batch_x, batch_y, batch_x_mark, batch_y_mark)

                    if self.args.model in STMIXER_V5_MODELS:
                        outputs, _ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    pred, true = self._get_pred_true(outputs, batch_y)

                    count += pred.reshape(-1).shape[0]
                    maes += sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                    mses += sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                    del pred
                    del true
                print(f'dataset{ii + 1} has loaded')
        maes = maes / count
        mses = mses / count
        print('mse:{}, mae:{}'.format(mses, maes))

        return self.model


def regularization(model, weight_decay, p=2):
    reg_loss = sum(
        torch.norm(param, p=p) ** 2
        for name, param in model.state_dict().items()
        if 'weight' in name and 'norm' not in name
    )
    return weight_decay * reg_loss
