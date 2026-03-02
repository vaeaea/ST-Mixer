import os
import torch
import numpy as np
import torch.nn as nn
import time
class Exp_Basic(object):
    def __init__(self, args):
        print("start")
        self.args = args
        start = time.time()


        if len(self.args.gpu)>1:
            self.device = self._acquire_device()
            device_ids = self.args.gpu #必须从零开始(这里0表示第1块卡，1表示第2块卡.)
            model = nn.DataParallel(self._build_model(), device_ids=device_ids)
            self.model = model.to(self.device)
        else:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)
        print(time.time()-start)
        #self.model = self._build_model()
        print("over")
    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            if len(self.args.gpu)>1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)[1:-1] 
                device = torch.device('cuda:{}'.format(self.args.gpu[0]))
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu[0]) 
                device = torch.device('cuda:{}'.format(self.args.gpu[0]))

            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
