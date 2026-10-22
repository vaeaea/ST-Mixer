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

#该文件主要负责模型在7个子数据集中训练验证测试时的 数据加载

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
            print("time data transform label embeding")
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
            df_stamp["year"] = df_stamp.date.dt.year
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            categorical_columns = ['year','month','day']
            categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
            df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
        elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
            print("time data transform value linear")
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
            df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
            df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
            df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5

        data_stamp = df_stamp.drop(['date'], 1).values
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
            print("time data transform label embeding")
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
            df_stamp["year"] = df_stamp.date.dt.year
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["minute"] = df_stamp.date.dt.minute
            categorical_columns = ['year','month','day','minute']
            categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
            df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
        elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
            print("time data transform value linear")
            df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
            df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
            df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
            df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
            df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
            df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5

        data_stamp = df_stamp.drop(['date'], 1).values
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
    
    
#weather\traffic\electricity数据集 ：7/1/2   
class Dataset_custom(Dataset):
    def __init__(self, flag='train', size=None,
                 features='M', data_path=r'‪traffic.csv',
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
                print("load traffic dataset")
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
                    print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    categorical_columns = ['year','month','day']
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2016)/2.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5

                data_stamp = df_stamp.drop(['date'], 1).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp
                
            elif self.data_set=='weather':
                print("load weather dataset")
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
                    print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["minute"] = df_stamp.date.dt.minute
                    categorical_columns = ['year','month','day','minute']
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2020)/1.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["minute"] = df_stamp.date.dt.minute / 59.0 - 0.5
                
                data_stamp = df_stamp.drop(['date'], 1).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp
          
            elif self.data_set=='electricity':
                print("load electricity dataset")
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
                    print("time data transform label embeding")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek
                    df_stamp["year"] = df_stamp.date.dt.year
                    df_stamp["month"] = df_stamp.date.dt.month
                    df_stamp["day"] = df_stamp.date.dt.day
                    df_stamp["dayofyear"] = df_stamp.date.dt.dayofyear
                    df_stamp["week"] = df_stamp.date.dt.isocalendar().week
                    categorical_columns = ['year','month','day','dayofyear','week']
                    categorical_scalers = fit_preprocessing(df_stamp, categorical_columns)
                    df_stamp = transform_inputs(df_stamp, categorical_scalers, categorical_columns)
                    
                elif self.time_data=='linear':  #将时间变量转变为-0.5,0.5之间的连续值，便于linear处理
                    print("time data transform value linear")
                    df_stamp["HourOfDay"] = df_stamp.date.dt.hour / 23.0 - 0.5 
                    df_stamp["DayOfWeek"] = df_stamp.date.dt.dayofweek / 6.0 - 0.5
                    df_stamp["year"] = (df_stamp.date.dt.year-2016)/3.0-0.5
                    df_stamp["month"] = (df_stamp.date.dt.month-1) / 11.0 - 0.5
                    df_stamp["day"] = (df_stamp.date.dt.day - 1) / 30.0 - 0.5  
                    df_stamp["dayofyear"] = (df_stamp.date.dt.dayofyear - 1) / 365.0 - 0.5  
                    df_stamp["week"] = (df_stamp.date.dt.week - 1) / 52.0 - 0.5 
                  
                data_stamp = df_stamp.drop(['date'], 1).values
                self.data_x = data[border1:border2]
                self.data_y = data[border1:border2]
                self.data_stamp = data_stamp  
                
          
                  
                data_stamp = df_stamp.drop(['date'], 1).values
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
    'weather': Dataset_custom,
    'electricity': Dataset_custom,
    'traffic': Dataset_custom,
}
 
        
def data_provider(args, flag):
    Data = data_dict[args.data_set] 
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size    

    data_set = Data( 
        data_path=args.data_path,
        flag=flag,
        size=[args.decomp_len, args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_set=args.data_set,
        time_data=args.time_data)
    print(flag, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader    


########################################################################################################    
def transform_inputs(df, categorical_scalers, categorical_columns):
    out = df.copy()

    for col in categorical_columns:
        string_df = df[col].apply(str)
        out[col] = categorical_scalers[col].transform(string_df)

    return out

def fit_preprocessing(train, categorical_columns): 
    categorical_scalers = {}
    num_classes = []
    for col in categorical_columns:
        srs = train[col].apply(str) 
        categorical_scalers[col] = LabelEncoder().fit(srs.values)
        num_classes.append(srs.nunique()) #统计每一列属性各自有多少个不同值。
    print(num_classes)
    return categorical_scalers 