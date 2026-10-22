import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler 
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore') 

#该文件主要负责模型在LTS大型数据集中训练验证测试时的 数据加载
    
#LTS数据集训练集的加载，需要用到Data文件夹下的文件，该部分文件受大小限制，需要从https://drive.google.com/file/d/1IM3ObGDeXeu39Pcf_0TwD3lbai7JiR3t/view?usp=drive_link处进行下载解压    
class Dataset_custom_train(Dataset):
    def __init__(self, flag='train', size=None,):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        elif size[0]<=size[1]: 
            self.seq_len = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]
        elif size[0]>size[1]: 
            self.seq_len = size[0]
            self.label_len = size[2]
            self.pred_len = size[3]
        # init
        assert flag in ['train']

        self.__read_data__()

    def __read_data__(self): 
        data_files = {
            'traffic': ("Data/traffic_raw.npz", int(17544*0.7)),
            'electricity': ("Data/electricity_raw.npz", int(26304*0.7)),
            'weather': ("Data/weather_raw.npz", int(52696*0.7)),
            'etth1': ("Data/etth1_raw.npz", 12 * 30 * 24),
            'etth2': ("Data/etth2_raw.npz", 12 * 30 * 24),
            'ettm1': ("Data/ettm1_raw.npz", 12 * 30 * 24 * 4),
            'ettm2': ("Data/ettm2_raw.npz", 12 * 30 * 24 * 4),
            'solar': ("Data/solar_raw.npz", int(52560*0.7)),
            'weather2': ("Data/weather2_raw.npz", int(8783*0.7)),
            'ett2m1': ("Data/ett2m1_raw.npz", 12 * 30 * 24),
            'ett2m2': ("Data/ett2m2_raw.npz", 12 * 30 * 24),
            'solar2': ("Data/solar2_raw.npz", int(8760*0.7)),
            'traffic2': ("Data/traffic2_raw.npz", int(48384*0.7)),
            'traffic3': ("Data/traffic3_raw.npz", int(16128*0.7)),
            'traffic4': ("Data/traffic4_raw.npz", int(24192*0.7)),
        }
        
        all_set = []
        for name, (path, train_len) in data_files.items():
            data = np.load(path)["arr_0"][:, :train_len, :]
            all_set.extend([data[i] for i in range(data.shape[0])])
            
        self.all_set = all_set
        
    def __getitem__(self, index): #表示选择一个变量进行样本采集
        #print(index)
        sampled_timeseries = np.array(self.all_set[index])

        sampled_point = np.random.randint(sampled_timeseries.shape[0]-self.pred_len-self.seq_len, size=1) 
        
        s_begin = sampled_point[0]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
  
        seq_x = np.array(sampled_timeseries)[s_begin:s_end, 0:1]
        seq_y = sampled_timeseries[r_begin + self.label_len:r_end, 0:1]
        seq_x_mark = sampled_timeseries[s_begin:s_end, 1:]
        seq_y_mark = sampled_timeseries[r_begin:r_end, 1:]

        return seq_x, seq_y, seq_x_mark, seq_y_mark        

    def __len__(self): 
        return 862+321+21+7+7+7+7+137 + 7+7+137+21 + 56+56+56
###########################################################################################################
#ETTh数据集：6/2/2
class Dataset_ETT_hour(Dataset):
    def __init__(self, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 scale=True, data_set='ETTh1', time_data='linear'):
        # size [args.decomp_len, args.seq_len, args.label_len, args.pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        elif size[0]<=size[1]: 
            self.seq_len = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]
        elif size[0]>size[1]: 
            self.seq_len = size[0]
            self.label_len = size[2]
            self.pred_len = size[3]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.data_set = data_set
        self.data_path = data_path
        self.time_data = time_data
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' :
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data] 

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
            df_stamp["year"] = df_stamp.date.dt.year
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
            df_stamp["week"] = df_stamp.date.dt.isocalendar().week
            df_stamp["minute"] = df_stamp.date.dt.minute
            categorical_columns = ['year','month','day','dayofyear','week',"minute"]
            categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
            df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)

        elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
            df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
            df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
            df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
            df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
            df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
            df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
        data_stamp = df_stamp.drop(columns=['date']).values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin + self.label_len:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

#ETTm数据集：6/2/2
class Dataset_ETT_minute(Dataset):
    def __init__(self, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 scale=True, data_set='ETTh1', time_data='linear'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        elif size[0]<=size[1]: 
            self.seq_len = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]
        elif size[0]>size[1]: 
            self.seq_len = size[0]
            self.label_len = size[2]
            self.pred_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.data_set = data_set
        self.data_path = data_path
        self.time_data = time_data
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' :
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data] 

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
 
            
        if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
            df_stamp["year"] = df_stamp.date.dt.year
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
            df_stamp["week"] = df_stamp.date.dt.isocalendar().week
            df_stamp["minute"] = df_stamp.date.dt.minute
            categorical_columns = ['year','month','day','dayofyear','week',"minute"]
            categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
            df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)

        elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
            df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
            df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
            df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
            df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
            df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
            df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                    
        data_stamp = df_stamp.drop(columns=['date']).values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin + self.label_len:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
#weather/traffic/electricity/solar及其衍生数据集 ：7/1/2      
class Dataset_custom(Dataset):
    def __init__(self, flag='train', size=None,
                 features='M', data_path=r'‪./Data/traffic.csv',
                 scale=True, data_set='traffic', time_data='linear'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        elif size[0]<=size[1]: 
            self.seq_len = size[1]
            self.label_len = size[2]
            self.pred_len = size[3]
        elif size[0]>size[1]: 
            self.seq_len = size[0]
            self.label_len = size[2]
            self.pred_len = size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.scale = scale
        self.data_set = data_set
        self.data_path = data_path
        self.time_data = time_data
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        if self.data_set=='season':
            df_raw = pd.read_csv(self.data_path, index_col=0)
        elif self.data_set=='trend':
            df_raw = pd.read_csv(self.data_path, index_col=0)
        else:
            df_raw = pd.read_csv(self.data_path)
        

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove('date')   #cols==other features
        df_raw = df_raw[['date'] + cols]  #对df_raw中的数据进行排序
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]   #{0,0.7,0.9}
        border2s = [num_train, num_train + num_vali, len(df_raw)]    #{0.7,0.9,1}
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
    
        if self.features == 'M':
            if self.data_set=='traffic':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 

                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                                        
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp
             
                
                
            elif self.data_set=='weather':
                #print("load weather dataset")
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 
                    
                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2020)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                    
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp
                             
 
            elif self.data_set=='electricity':
                #print("load electricity dataset")
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                
                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2016)/3.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5
                    
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp  

            elif self.data_set=='solar':
                #print("load solar dataset")
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                
                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = df_stamp.date.dt.year-2016
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5
                    
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp  
 
            elif self.data_set=='weather2':
                #print("load weather2 dataset")
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 
                    
                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2020)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                    
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp

            elif self.data_set=='solar2':
                #print("load solar2 dataset")
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                
                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = df_stamp.date.dt.year-2016
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5
                    
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp  
                
            elif self.data_set=='traffic2':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 

                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2017)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                                        
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp      
                
            elif self.data_set=='traffic3':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 

                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2017)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                                        
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp                     
         
            elif self.data_set=='traffic4':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]  #多变量  ：除时间外的数据  df_raw==》df_data+df_stamp

                if self.scale:   #按照训练集进行标准化
                    train_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data.values)
                    data = self.scaler.transform(df_data.values)
                else:
                    data = df_data.values  

                #提取时间特征
                df_stamp = df_raw[['date']][border1:border2]
                df_stamp['date'] = pd.to_datetime(df_stamp.date) 

                if self.time_data=='embeding':  #将时间变量转变为0，1，2，3，的类别项，便于embeding处理
                    #print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','dayofyear','week',"minute"]
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    #print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2017)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5  
                                        
                data_stamp = df_stamp.drop(columns=['date']).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp        
        
    def __getitem__(self, index):
        if self.features == 'M':
            #print(index)
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin + self.label_len:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            return seq_x, seq_y, seq_x_mark, seq_y_mark        
          
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
 

    
#########################################################################################################    
 
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ETT2m1': Dataset_ETT_hour,
    'ETT2m2': Dataset_ETT_hour,    
    
    'weather': Dataset_custom,
    'electricity': Dataset_custom,
    'traffic': Dataset_custom,  
    'solar': Dataset_custom,  
    'solar2': Dataset_custom,  
    'weather2': Dataset_custom,
    'traffic2': Dataset_custom, 
    'traffic3': Dataset_custom, 
    'traffic4': Dataset_custom, 
    'all': Dataset_custom_train, 
}

def data_provider(args, flag): 
    if flag == 'train': 
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  

        Data_all = data_dict['all']  
        data_set_all = Data_all( 
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],)   

        data_loader_all = DataLoader(
            data_set_all,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False) 
        return data_set_all, data_loader_all  
    
    elif flag == 'val':
        shuffle_flag = True
        drop_last = False
        batch_size = 1
        Data_ETTh1 = data_dict['ETTh1'] 
        Data_ETTh2 = data_dict['ETTh2'] 
        Data_ETTm1 = data_dict['ETTm1'] 
        Data_ETTm2 = data_dict['ETTm2'] 
        Data_weather = data_dict['weather'] 
        Data_electricity = data_dict['electricity'] 
        Data_traffic = data_dict['traffic'] 
        Data_solar = data_dict['solar']
        Data_ETT2m1 = data_dict['ETT2m1'] 
        Data_ETT2m2 = data_dict['ETT2m2']         
        Data_solar2 = data_dict['solar2']
        Data_weather2 = data_dict['weather2'] 
        Data_traffic2 = data_dict['traffic2'] 
        Data_traffic3 = data_dict['traffic3'] 
        Data_traffic4 = data_dict['traffic4'] 
        
        data_set_ETTh1 = Data_ETTh1( 
            data_path=r'./all_six_datasets/ETT-small/ETTh1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTh1',
            time_data=args.time_data) 
        data_set_ETTh2 = Data_ETTh2( 
            data_path=r'./all_six_datasets/ETT-small/ETTh2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTh2',
            time_data=args.time_data) 
        data_set_ETTm1 = Data_ETTm1( 
            data_path=r'./all_six_datasets/ETT-small/ETTm1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTm1',
            time_data=args.time_data) 
        data_set_ETTm2 = Data_ETTm2( 
            data_path=r'./all_six_datasets/ETT-small/ETTm2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTm2',
            time_data=args.time_data) 
        data_set_weather = Data_weather( 
            data_path=r'./all_six_datasets/weather/weather.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='weather',
            time_data=args.time_data) 
        data_set_electricity = Data_electricity( 
            data_path=r'./all_six_datasets/electricity/electricity.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='electricity',
            time_data=args.time_data) 
        data_set_traffic = Data_traffic( 
            data_path=r'./all_six_datasets/traffic/traffic.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic',
            time_data=args.time_data)  
        data_set_solar = Data_solar( 
            data_path=r'./all_six_datasets/solar/solar.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='solar',
            time_data=args.time_data)  
        data_set_ETT2m1 = Data_ETT2m1( 
            data_path=r'./all_six_datasets/ETT-small/ETT2m1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETT2m1',
            time_data=args.time_data) 
        data_set_ETT2m2 = Data_ETT2m2( 
            data_path=r'./all_six_datasets/ETT-small/ETT2m2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETT2m2',
            time_data=args.time_data)         
        data_set_solar2 = Data_solar2( 
            data_path=r'./all_six_datasets/solar/solar2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='solar2',
            time_data=args.time_data)  
        data_set_weather2 = Data_weather2( 
            data_path=r'./all_six_datasets/weather/weather2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='weather2',
            time_data=args.time_data)         
        data_set_traffic2 = Data_traffic2( 
            data_path=r'./all_six_datasets/traffic/traffic2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic2',
            time_data=args.time_data)          
        data_set_traffic3 = Data_traffic3( 
            data_path=r'./all_six_datasets/traffic/traffic3.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic3',
            time_data=args.time_data)          
        data_set_traffic4 = Data_traffic4( 
            data_path=r'./all_six_datasets/traffic/traffic4.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic4',
            time_data=args.time_data)  
        
        
        data_loader_ETTh1 = DataLoader(
            data_set_ETTh1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTh2 = DataLoader(
            data_set_ETTh2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTm1 = DataLoader(
            data_set_ETTm1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTm2 = DataLoader(
            data_set_ETTm2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_weather = DataLoader(
            data_set_weather,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_electricity = DataLoader(
            data_set_electricity,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic = DataLoader(
            data_set_traffic,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_solar = DataLoader(
            data_set_solar,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETT2m1 = DataLoader(
            data_set_ETT2m1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETT2m2 = DataLoader(
            data_set_ETT2m2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_weather2 = DataLoader(
            data_set_weather2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_solar2 = DataLoader(
            data_set_solar2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic2 = DataLoader(
            data_set_traffic2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic3 = DataLoader(
            data_set_traffic3,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic4 = DataLoader(
            data_set_traffic4,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        
        return [data_set_ETTh1, data_set_ETTh2,data_set_ETTm1,data_set_ETTm2,data_set_weather,data_set_electricity,data_set_traffic,data_set_solar,data_set_weather2,data_set_ETT2m1,data_set_ETT2m2,data_set_solar2,data_set_traffic2,data_set_traffic3,data_set_traffic4],[data_loader_ETTh1,data_loader_ETTh2,data_loader_ETTm1,data_loader_ETTm2,data_loader_weather,data_loader_electricity,data_loader_traffic,data_loader_solar,data_loader_weather2,data_loader_ETT2m1,data_loader_ETT2m2,data_loader_solar2,data_loader_traffic2,data_loader_traffic3,data_loader_traffic4] 
        
    elif flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data_ETTh1 = data_dict['ETTh1'] 
        Data_ETTh2 = data_dict['ETTh2'] 
        Data_ETTm1 = data_dict['ETTm1'] 
        Data_ETTm2 = data_dict['ETTm2'] 
        Data_weather = data_dict['weather'] 
        Data_electricity = data_dict['electricity'] 
        Data_traffic = data_dict['traffic'] 
        Data_solar = data_dict['solar'] 
        Data_ETT2m1 = data_dict['ETT2m1'] 
        Data_ETT2m2 = data_dict['ETT2m2']         
        Data_solar2 = data_dict['solar2']
        Data_weather2 = data_dict['weather2']         
        Data_traffic2 = data_dict['traffic2'] 
        Data_traffic3 = data_dict['traffic3'] 
        Data_traffic4 = data_dict['traffic4']         
        
        data_set_ETTh1 = Data_ETTh1( 
            data_path=r'./all_six_datasets/ETT-small/ETTh1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTh1',
            time_data=args.time_data) 
        data_set_ETTh2 = Data_ETTh2( 
            data_path=r'./all_six_datasets/ETT-small/ETTh2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTh2',
            time_data=args.time_data) 
        data_set_ETTm1 = Data_ETTm1( 
            data_path=r'./all_six_datasets/ETT-small/ETTm1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTm1',
            time_data=args.time_data) 
        data_set_ETTm2 = Data_ETTm2( 
            data_path=r'./all_six_datasets/ETT-small/ETTm2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETTm2',
            time_data=args.time_data) 
        data_set_weather = Data_weather( 
            data_path=r'./all_six_datasets/weather/weather.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='weather',
            time_data=args.time_data) 
        data_set_electricity = Data_electricity( 
            data_path=r'./all_six_datasets/electricity/electricity.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='electricity',
            time_data=args.time_data) 
        data_set_traffic = Data_traffic( 
            data_path=r'./all_six_datasets/traffic/traffic.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic',
            time_data=args.time_data)  
        data_set_solar = Data_solar( 
            data_path=r'./all_six_datasets/solar/solar.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='solar',
            time_data=args.time_data)  
        data_set_ETT2m1 = Data_ETT2m1( 
            data_path=r'./all_six_datasets/ETT-small/ETT2m1.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETT2m1',
            time_data=args.time_data) 
        data_set_ETT2m2 = Data_ETT2m2( 
            data_path=r'./all_six_datasets/ETT-small/ETT2m2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='ETT2m2',
            time_data=args.time_data)         
        data_set_solar2 = Data_solar2( 
            data_path=r'./all_six_datasets/solar/solar2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='solar2',
            time_data=args.time_data)  
        data_set_weather2 = Data_weather2( 
            data_path=r'./all_six_datasets/weather/weather2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='weather2',
            time_data=args.time_data)         
        data_set_traffic2 = Data_traffic2( 
            data_path=r'./all_six_datasets/traffic/traffic2.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic2',
            time_data=args.time_data)          
        data_set_traffic3 = Data_traffic3( 
            data_path=r'./all_six_datasets/traffic/traffic3.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic3',
            time_data=args.time_data)          
        data_set_traffic4 = Data_traffic4( 
            data_path=r'./all_six_datasets/traffic/traffic4.csv',
            flag=flag,
            size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
            features=args.features ,
            data_set='traffic4',
            time_data=args.time_data)      
    
    
    
    
    
    
        data_loader_ETTh1 = DataLoader(
            data_set_ETTh1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTh2 = DataLoader(
            data_set_ETTh2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTm1 = DataLoader(
            data_set_ETTm1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETTm2 = DataLoader(
            data_set_ETTm2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_weather = DataLoader(
            data_set_weather,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_electricity = DataLoader(
            data_set_electricity,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic = DataLoader(
            data_set_traffic,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_solar = DataLoader(
            data_set_solar,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)  
        data_loader_ETT2m1 = DataLoader(
            data_set_ETT2m1,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_ETT2m2 = DataLoader(
            data_set_ETT2m2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)        
        data_loader_weather2 = DataLoader(
            data_set_weather2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)        
        data_loader_solar2 = DataLoader(
            data_set_solar2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)         
        data_loader_traffic2 = DataLoader(
            data_set_traffic2,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic3 = DataLoader(
            data_set_traffic3,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)
        data_loader_traffic4 = DataLoader(
            data_set_traffic4,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if args.num_workers > 0 else False)        
        
        
        return [data_set_ETTh1, data_set_ETTh2,data_set_ETTm1,data_set_ETTm2,data_set_weather,data_set_electricity,data_set_traffic,data_set_solar,data_set_weather2,data_set_ETT2m1,data_set_ETT2m2,data_set_solar2,data_set_traffic2,data_set_traffic3,data_set_traffic4],[data_loader_ETTh1,data_loader_ETTh2,data_loader_ETTm1,data_loader_ETTm2,data_loader_weather,data_loader_electricity,data_loader_traffic,data_loader_solar,data_loader_weather2,data_loader_ETT2m1,data_loader_ETT2m2,data_loader_solar2,data_loader_traffic2,data_loader_traffic3,data_loader_traffic4]  

########################################################################################################    
 ########################################################################################################    
def transform_inputs(df, categorical_scalers, categorical_columns):
    out = df.copy()
    for col in categorical_columns:
        out[col] = categorical_scalers[col].transform(df[col].astype(str))
    return out

def fit_preprocessing(train, categorical_columns): 
    categorical_scalers = {}
    num_classes = []
    for col in categorical_columns:
        srs = train[col].astype(str)
        categorical_scalers[col] = LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique())
    print(num_classes)
    return categorical_scalers 