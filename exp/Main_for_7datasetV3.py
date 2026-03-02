import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim 
from exp.Main_basic import Exp_Basic 
from Model.STMixer import STMixer 
from Model.Dlinear import Dlinear
from Model.Nlinear import Nlinear
from Model.Linear import Linear
from Model.patchtst import patchtst 
from Model.STMixer_abla_1 import STMixer_abla_1
from Model.STMixer_abla_2 import STMixer_abla_2
from Model.STMixer_abla_3 import STMixer_abla_3 
from Model.STMixer_abla_1_v3 import STMixer_abla_1_v3
from Model.STMixer_abla_2_v3 import STMixer_abla_2_v3
from Model.STMixer_abla_3_v3 import STMixer_abla_3_v3
from Model.STMixer_abla_1_v5 import STMixer_abla_1_v5
from Model.STMixer_abla_2_v5 import STMixer_abla_2_v5
from Model.STMixerv2 import STMixerv2
from Model.STMixerv3 import STMixerv3
from Model.STMixerv4 import STMixerv4
from Model.STMixerv5 import STMixerv5
from Model.Autoformer import Model as Autoformer
from Model.FEDformer import Model as Fedformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch.utils.data import DataLoader
from Data.Data_load_7dataV3 import *
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


#该文件用于模型在大型LTS数据集中训练验证测试

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        

    def _build_model(self):
        print("building model:",self.args.model)
        #start_model = time.time()
        if self.args.model=='STMixer':
            model = STMixer(self.args) 
        elif self.args.model=='Dlinear':
            model = Dlinear(self.args)        
        elif self.args.model=='Nlinear':
            model = Nlinear(self.args) 
        elif self.args.model=='Linear':
            model = Linear(self.args) 
        elif self.args.model=='patchtst':
            model = patchtst(self.args)  
        elif self.args.model=='STMixer_abla_1': #消融实验模型1
            model = STMixer_abla_1(self.args)  
        elif self.args.model=='STMixer_abla_2': #消融实验模型2
            model = STMixer_abla_2(self.args)  
        elif self.args.model=='STMixer_abla_3': #消融实验模型3
            model = STMixer_abla_3(self.args)   
        elif self.args.model=='STMixer_abla_1_v3': #消融实验模型1
            model = STMixer_abla_1_v3(self.args)  
        elif self.args.model=='STMixer_abla_2_v3': #消融实验模型2
            model = STMixer_abla_2_v3(self.args)  
        elif self.args.model=='STMixer_abla_3_v3': #消融实验模型3
            model = STMixer_abla_3_v3(self.args)   
        elif self.args.model=='STMixer_abla_1_v5': #消融实验模型1
            model = STMixer_abla_1_v5(self.args)  
        elif self.args.model=='STMixer_abla_2_v5': #消融实验模型1
            model = STMixer_abla_2_v5(self.args)  
        elif self.args.model=='STMixerv2': 
            model = STMixerv2(self.args)    
        elif self.args.model=='STMixerv3': 
            model = STMixerv3(self.args)   
        elif self.args.model=='STMixerv4': 
            model = STMixerv4(self.args)  
        elif self.args.model=='STMixerv5': 
            model = STMixerv5(self.args)
        elif self.args.model=='Autoformer':
            model = Autoformer(self.args)
        elif self.args.model=='Fedformer':
            model = Fedformer(self.args)
        #print(time.time()-start_model)
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

    def _select_criterion(self): #该部分使用了MAE作为损失函数
        print("selecting criterion")
        if self.args.loss == 'mse':
            criterion = nn.MSELoss()
            return criterion
        if self.args.loss == 'mae': 
            criterion = nn.L1Loss()  
            return criterion        
        if self.args.loss == 'all': 
            criterion_mae = nn.L1Loss()  
            criterion_mse = nn.MSELoss()  
            return criterion_mae,criterion_mse
        

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
  
 
        max_allocated_memory = 0 
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
            
            for _ in range(self.args.report_freq):
                for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
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


                    loss.backward()
                    model_optim.step()
                
                    # 获取当前GPU内存分配情况  
                    # memory_map = get_gpu_memory_map(device_index=self.args.gpu[0])  
                    # current_allocated = memory_map['allocated']  
                    # 更新最大分配内存  
                    # if current_allocated > max_allocated_memory:  
                    #     max_allocated_memory = current_allocated  
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
            print('iter time',iter_count/(time.time()-epoch_time))
            train_loss = np.average(train_loss)
            train_loss_v = np.average(train_loss_v)
            if epoch%self.args.val_time == 0:
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
            # nni.report_intermediate_result(vali_loss)
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print("train over")
        # nni.report_final_result(np.min(np.array(min_val_loss)))
        return self.model      
    
    def vali(self, vali_loader, criterion):
        #print("valval")
        # preds = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # trues = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        preds = []
        trues = []
        maes = []
        mses = []
        maes = 0
        mses = 0
        count = 0
        self.model.eval()
        # 创建一个FlopCountAnalysis实例  
        fff_time = time.time()
        if 'STMixer' in str(self.model):
            input_tensor1 = torch.randn(256, self.args.decomp_len, 1).to(self.device)
            input_tensor2 = torch.randn(256, self.args.decomp_len, self.args.mark_size).to(self.device)  
            input_tensor3 = torch.randn(256, self.args.pred_len, 1).to(self.device)  
            input_tensor4 = torch.randn(256, self.args.pred_len+self.args.label_len, self.args.mark_size).to(self.device)  
        
        else:
            input_tensor1 = torch.randn(256, self.args.seq_len, 1).to(self.device)
            input_tensor2 = torch.randn(256, self.args.seq_len, self.args.mark_size).to(self.device)  
            input_tensor3 = torch.randn(256, self.args.pred_len, 1).to(self.device)  
            input_tensor4 = torch.randn(256, self.args.pred_len+self.args.label_len, self.args.mark_size).to(self.device)  
        max_memory_used_test = 0
        torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])  
        for _ in range(100):
            flops = FlopCountAnalysis(self.model,(input_tensor1,input_tensor2,input_tensor3,input_tensor4) )  
            current_max_memory_test = torch.cuda.max_memory_allocated(device=self.args.gpu[0])  
            if current_max_memory_test > max_memory_used_test:  
                max_memory_used_test = current_max_memory_test   
        cost_time = (time.time() - fff_time)/100
        print(f"Total vali FLOPs: {flops.total()}","FPS: {}".format(1/cost_time))    
        print(f"Maximum GPU memory allocated during training: {max_memory_used_test / (1024**2):.2f} MB")  
  
 
        max_allocated_memory = 0 
        max_memory_used = 0
        # 训练前重置最大内存分配记录  
        torch.cuda.reset_max_memory_allocated(device=self.args.gpu[0])  
        
        with torch.no_grad():
            for ii in range(8+7):  #共15个子数据集
                for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader[ii]): 
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                    if self.args.model in ['STMixerv5','STMixer_abla_1_v5']:
                        outputs, end_paddings_list = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                        if self.args.pred_len==720:  #预测长度为720时，内存不足......
                            pred = outputs.detach().cpu().numpy().astype(np.float16)
                            true = batch_y.detach().cpu().numpy().astype(np.float16) 
                        else:                    
                            pred = outputs.detach().cpu().numpy()
                            true = batch_y.detach().cpu().numpy()  
                        count += pred.reshape(-1).shape[0]
 

                        if self.args.loss == 'mae': 
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += sum(MAE11(end_paddings_list[i].detach().cpu().numpy().reshape(-1), true[:,:end_paddings_list[i].shape[1],:].reshape(-1)) )
                            mae = sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                            maes += mae
                            maes += end_padding_loss/len(end_paddings_list)
                        elif self.args.loss == 'mse': 
                            end_padding_loss = 0
                            for i in range(len(end_paddings_list)):
                                end_padding_loss += sum(MSE11(end_paddings_list[i].detach().cpu().numpy().reshape(-1), true[:,:end_paddings_list[i].shape[1],:].reshape(-1)) )
                            mse = sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                            mses += mse 
                            mses += end_padding_loss/len(end_paddings_list)
                        del pred
                        del true

                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                
                        if self.args.pred_len==720:  #预测长度为720时，内存不足......
                            pred = outputs.detach().cpu().numpy().astype(np.float16)
                            true = batch_y.detach().cpu().numpy().astype(np.float16)
                        else:                    
                            pred = outputs.detach().cpu().numpy()
                            true = batch_y.detach().cpu().numpy() 
                        count += pred.reshape(-1).shape[0]
 

                        if self.args.loss == 'mae': 
                            mae = sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                            maes += mae
                        elif self.args.loss == 'mse': 
                            mse = sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                            mses += mse
                        del pred
                        del true 
                    current_max_memory = torch.cuda.max_memory_allocated(device=self.args.gpu[0])  
                    if current_max_memory > max_memory_used:  
                        max_memory_used = current_max_memory  
                # print(f'{ii}个数据集vali完毕')
        print(f"Maximum GPU memory allocated during valiing: {max_memory_used / (1024**2):.2f} MB")  
        if self.args.loss == 'mae': 
            maes = maes/count   
            self.model.train()
            return maes   
        elif self.args.loss == 'mse':    
            mses = mses/count
            self.model.train()
            return mses    

    def test(self, setting, test=0):
        #print("testtest")
        test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints , setting, 'checkpoint.pth')))

        # preds = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        # trues = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        preds = []
        trues = [] 
        mses = []
        maes = [] 
        mses = 0
        maes = 0
        count = 0
        folder_path = './test_results_picts/' + self.args.data_set + '/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        print("start test")
        if not os.path.exists('./all_pred_result/' + self.args.model + self.args.data_set + '/'):
            os.makedirs('./all_pred_result/' + self.args.model + self.args.data_set + '/')
            os.makedirs('./all_true_result/' + self.args.model + self.args.data_set + '/')
            os.makedirs('./all_mse_result/' + self.args.model + self.args.data_set + '/')
            os.makedirs('./all_mae_result/' + self.args.model + self.args.data_set + '/')
        # f_720_pred = open('./all_pred_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        # f_720_true = open('./all_true_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        f_720_mse = open('./all_mse_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        f_720_mae = open('./all_mae_result/' + self.args.model + self.args.data_set + '/' + setting + '.txt', 'a')
        with torch.no_grad():
            for ii in range(8+7):
                
                for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader[ii]):
                     
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                    if self.args.model in ['STMixerv5','STMixer_abla_1_v5']:
                        outputs,_ = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
                    if self.args.pred_len==720:
                        pred = outputs.detach().cpu().numpy().astype(np.float16)
                        true = batch_y.detach().cpu().numpy().astype(np.float16)
                    else:                    
                        pred = outputs.detach().cpu().numpy()
                        true = batch_y.detach().cpu().numpy()

                    # preds[ii].append(pred)
                    # trues[ii].append(true)
                    # preds += list(pred.reshape(-1))
                    # trues += list(true.reshape(-1))
                    # mae, mse, _, _, _, _ = metric(pred.reshape(-1), true.reshape(-1))
                    # mae = MAE11(pred.reshape(-1), true.reshape(-1))
                    # mse = MSE11(pred.reshape(-1), true.reshape(-1))
                    count += pred.reshape(-1).shape[0]
                    maes += sum(MAE11(pred.reshape(-1), true.reshape(-1)))
                    mses += sum(MSE11(pred.reshape(-1), true.reshape(-1)))
                    # maes += list(mae.reshape(-1))
                    # mses += list(mse.reshape(-1))
                    # f_720_pred.write(str(list(pred.reshape(-1))))
                    # f_720_true.write(str(list(true.reshape(-1))))
                    # f_720_mse.write(str(list(mse.reshape(-1))))
                    # f_720_mae.write(str(list(mae.reshape(-1)))) 
                    # f_720_mse.writelines(f'{value}\n' for value in mse.reshape(-1))
                    # f_720_mae.writelines(f'{value}\n' for value in mae.reshape(-1))
                    # if i % 20 == 0:
                    #     input = batch_x.detach().cpu().numpy()
                    #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    del pred
                    del true
                print(f'dataset{ii+1} has loaded')
        # maes = np.mean(np.array(maes))
        # mses = np.mean(np.array(mses))
        # print('mse:{}, mae:{}'.format(mses, maes))
        maes = maes/count
        mses = mses/count
        print('mse:{}, mae:{}'.format(mses, maes))


        
        # # preds_all = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None] 
        # # trues_all = [None,None,None,None,None,None,None,None,None,None,None,None,None,None,None] 
        # # for ii in range(8+7):
        # #     preds_all[ii] = np.concatenate(preds[ii],axis=0)
        # #     trues_all[ii] = np.concatenate(trues[ii],axis=0)  #(BATCH,TIME_LEN,DIM)
        # #     preds_all[ii] = preds_all[ii].reshape(-1)
        # #     trues_all[ii] = trues_all[ii].reshape(-1)
        # # del preds
        # # del trues
        # # preds = np.concatenate([preds_all[0],preds_all[1],preds_all[2],preds_all[3],preds_all[4],preds_all[5],preds_all[6],preds_all[7],preds_all[8],preds_all[9],preds_all[10],preds_all[11],preds_all[12],preds_all[13],preds_all[14]],axis=0)
        # # del preds_all
        # # trues = np.concatenate([trues_all[0],trues_all[1],trues_all[2],trues_all[3],trues_all[4],trues_all[5],trues_all[6],trues_all[7],trues_all[8],trues_all[9],trues_all[10],trues_all[11],trues_all[12],trues_all[13],trues_all[14]],axis=0)
        # # del trues_all
        # preds = np.array(preds)
        # trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        # # result save
        # folder_path = './results_matrics/' + self.args.data_set+ '/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        # print('mse:{}, mae:{}, smape:{}'.format(mse, mae, smape))
        
        # txt = "result_dataset_{}_sl{}_pl{}.txt".format(self.args.data_set,self.args.seq_len,self.args.pred_len)
        # txt_path = './results_txt/' + self.args.data_set+ '/' + txt  
        # if not os.path.exists('./results_txt/' + self.args.data_set+ '/'):
        #     os.makedirs('./results_txt/' + self.args.data_set+ '/')
        # f = open(txt_path, 'a')
        # for eachArg, value in self.args.__dict__.items():
        #     f.write(eachArg + ' : ' + str(value) + '\n')
        # f.write('mse:{}, mae:{}, smape:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, smape, rmse, mape, mspe))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, smape]))
        # #np.save(folder_path + 'pred.npy', preds)
        # #np.save(folder_path + 'true.npy', trues)

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
