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
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import DataLoader
from Data.Data_load import *
from dateutil.relativedelta import relativedelta
import datetime
from fvcore.nn import FlopCountAnalysis
# import nni
warnings.filterwarnings('ignore')
def get_gpu_memory_map(device_index):  
    """获取当前GPU的内存使用情况映射（需要PyTorch 1.6.0或更高版本）"""  
    if torch.cuda.is_available():  
        allocated = torch.cuda.memory_allocated(device=device_index)  
        cached = torch.cuda.memory_cached(device=device_index)  
        return {'allocated': allocated, 'cached': cached}  
    else:  
        return {'allocated': 0, 'cached': 0}  
    
def MAE11(pred, true):
    return np.abs(pred - true)


def MSE11(pred, true):
    return (pred - true) ** 2
#该文件用于模型在7个子数据集中训练验证测试

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
    def _build_model(self):
        print("building model:",self.args.model)
        if self.args.model=='STMixerv5':
            model = STMixerv5(self.args)
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

    def train(self, setting):
        #print("12345")
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
        # 创建一个FlopCountAnalysis实例  
        fff_time = time.time()
        if 'STMixer' in str(self.model):
            input_tensor1 = torch.randn(self.args.batch_size, self.args.decomp_len, self.args.input_size).to(self.device)
            input_tensor2 = torch.randn(self.args.batch_size, self.args.decomp_len, self.args.mark_size).to(self.device)  
            input_tensor3 = torch.randn(self.args.batch_size, self.args.pred_len , self.args.input_size).to(self.device)  
            input_tensor4 = torch.randn(self.args.batch_size, self.args.pred_len+self.args.label_len, self.args.mark_size).to(self.device)  
        
        else:
            input_tensor1 = torch.randn(self.args.batch_size, self.args.seq_len, self.args.input_size).to(self.device)
            input_tensor2 = torch.randn(self.args.batch_size, self.args.seq_len, self.args.mark_size).to(self.device)  
            input_tensor3 = torch.randn(self.args.batch_size, self.args.pred_len , self.args.input_size).to(self.device)  
            input_tensor4 = torch.randn(self.args.batch_size, self.args.pred_len+self.args.label_len, self.args.mark_size).to(self.device)  
        max_memory_used_test = 0
        torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])  
        for _ in range(100):
            flops = FlopCountAnalysis(self.model,(input_tensor1,input_tensor2,input_tensor3,input_tensor4) )  
            current_max_memory_test = torch.cuda.max_memory_allocated(device=self.args.gpu[0])  
            if current_max_memory_test > max_memory_used_test:  
                max_memory_used_test = current_max_memory_test   
        cost_time = (time.time() - fff_time)/100
        print(f"Total Train FLOPs: {flops.total()}","FPS: {}".format(1/cost_time))    
        print(f"Maximum GPU memory allocated during training: {max_memory_used_test / (1024**2):.2f} MB")  

        max_memory_used = 0

        #print("start train")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_v = []
            self.model.train()
            epoch_time = time.time()
            # 训练前重置最大内存分配记录  
            torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])  
            for i,(batch_x, batch_y, batch_x_mark, batch_y_mark)  in enumerate(train_loader):
                #(batch_x, batch_y, batch_x_mark, batch_y_mark)（回顾窗口数据，预测时段数据，回顾窗口时间信息，预测时段时间信息）
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['STMixerv5','STMixer_abla_1_v5']:
                    outputs, end_paddings_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark) 
                    #正则化损失
                    if self.args.weight_decay>0:
                        reg_loss = regularization(self.model,self.args.weight_decay,self.args.p)
                        if self.args.loss == 'all': 
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += criterion[0](end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                                end_padding_loss += criterion[1](end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                            loss = criterion[0](outputs, batch_y.to(self.device)) + criterion[1](outputs, batch_y.to(self.device)) + reg_loss + end_padding_loss
                        else:
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += criterion(end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                            loss = criterion(outputs, batch_y.to(self.device)) + reg_loss + end_padding_loss
                    else:
                        if self.args.loss == 'all': 
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += criterion[0](end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                                end_padding_loss += criterion[1](end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                            loss = criterion[0](outputs, batch_y.to(self.device)) + criterion[1](outputs, batch_y.to(self.device)) + end_padding_loss/len(end_paddings_list)
                        else:
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += criterion(end_paddings_list[i], batch_y[:,:end_paddings_list[i].shape[1],:].to(self.device))
                            loss = criterion(outputs, batch_y.to(self.device)) + end_padding_loss/len(end_paddings_list)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark) 
                    #正则化损失
                    if self.args.weight_decay>0:
                        reg_loss = regularization(self.model,self.args.weight_decay,self.args.p)
                        if self.args.loss == 'all': 
                            loss = criterion[0](outputs, batch_y.to(self.device)) + criterion[1](outputs, batch_y.to(self.device)) + reg_loss
                        else:
                            loss = criterion(outputs, batch_y.to(self.device)) + reg_loss
                    else:
                        if self.args.loss == 'all': 
                            loss = criterion[0](outputs, batch_y.to(self.device)) + criterion[1](outputs, batch_y.to(self.device))
                        else:
                            loss = criterion(outputs, batch_y.to(self.device))
                train_loss.append(loss.item())
                if self.args.loss == 'all': 
                    train_loss_v.append((criterion[0](outputs, batch_y.to(self.device)) + criterion[1](outputs, batch_y.to(self.device))).item())  
                else:
                    train_loss_v.append(criterion(outputs, batch_y.to(self.device)).item())                

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
            total_trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'training parameters {total_trainable_params} ')        
            
            print("Epoch: {} train cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_loss_v = np.average(train_loss_v)
            if epoch%self.args.val_time == 0:
            
                vali_time = time.time()
                vali_loss = self.vali(vali_loader, criterion)
                print("Epoch: {} vali cost time: {}".format(epoch + 1, time.time() - vali_time))
                test_time = time.time()
                if epoch%5 == 1000:
                    test_loss = self.vali(test_loader, criterion) 
                else:
                    test_loss = vali_loss
                print("Epoch: {} test cost time: {}".format(epoch + 1, time.time() - test_time))
                
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Loss wo l2: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss, train_loss_v, vali_loss, test_loss))
                #没有nan再进行其他操作
                if np.isnan(train_loss):
                    break   
                min_val_loss.append(vali_loss)    
                early_stopping(vali_loss, self.model, path, epoch+1)           
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
                #(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['STMixerv5','STMixer_abla_1_v5']:
                    outputs, end_paddings_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    pred = outputs 
                    true = batch_y 

                    end_padding_loss = 0 
                    for i in range(len(end_paddings_list)): 
                        end_padding_loss += criterion(end_paddings_list[i]  , true[:,:end_paddings_list[i].shape[1],:] ) 
                    loss = criterion(pred, true) + end_padding_loss/len(end_paddings_list)
                    total_loss.append(loss.item())
                    del pred
                    del true
 
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
    

                    pred = outputs.detach().cpu()#(B,T,862)
                    true = batch_y.detach().cpu()

                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
                    del pred
                    del true
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def test(self, setting, test=0):
        #print("testtest")
        test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints , setting, 'checkpoint.pth')))

        mses = 0
        maes = 0
        count = 0
        folder_path = './test_results_picts/' + self.args.data_set + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print("start test")
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): 
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
 
 
                if self.args.model in ['STMixerv5','STMixer_abla_1_v5']:
                    outputs,_ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
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
                maes += sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                mses += sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                del pred
                del true
        maes = maes/count
        mses = mses/count
        print('mse:{}, mae:{}'.format(mses, maes))
        return self.model

def regularization(model,weight_decay,p=2):
    weight_list=[]
    for name, param in model.state_dict().items():
        if 'weight' in name:
            #if 'embed' not in name:
                if 'norm' not in name:
                    weight = (name, param)
                    weight_list.append(weight)
                    #print(name)
    reg_loss=0
    for name, w in weight_list:
        l2_reg = torch.norm(w, p=p)
        reg_loss = reg_loss + l2_reg**2

    reg_loss=weight_decay*reg_loss  
    return reg_loss
