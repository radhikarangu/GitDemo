# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:52:05 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Import CSV file into a dataframe
delhidata=pd.read_excel('D:\DS project Files\Delhi (1).xlsx')
#EDA    
#Index(['date', 'pm25'], dtype='object')
delhidata.head()
delhidata=delhidata.iloc[::-1]
delhidata.head()
delhidata.info()
delhidata.dtypes
delhidata['pm25']  = pd.to_numeric(delhidata['pm25'] ,errors='coerce')
delhidata.dtypes
delhidata.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
delhidata1 = pd.DataFrame({'date': pd.date_range('2018-01-01', '2018-04-21', freq='1H', closed='left')})
delhidata2 = delhidata1.iloc[:2617,:]
delhidata3 = pd.merge(delhidata,delhidata2,on='date',how='right') 
delhidata3.info()
delhidata3.sort_values("date", axis = 0, ascending = True,inplace = True, na_position ='last')
sns.heatmap(delhidata.isnull(),cbar=True)
delhidata3.tail()
delhidata3.isna().sum()
delhidata3.info()
delhidata3.set_index(['date'],inplace=True)
delhidata3.shape
delhidata3.isnull().sum()
delhidata3_time=delhidata3.interpolate(method='time')
delhidata3_time.plot()

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
hw=Holt(delhidata3_time["pm25"]).fit()
hw_pred=hw.predict(start=2100,end=2616)
hw_train=hw.predict(start=0,end=2100)
hw_rmse_train=np.sqrt(mean_squared_error(hw_train,delhidata3_time["pm25"].iloc[:2101]))
hw_rmse_train#52

hw_test=hw.predict(start=2102,end=2616)
hw_rmse_test=np.sqrt(mean_squared_error(hw_test,delhidata3_time["pm25"].iloc[2102:]))
hw_rmse_test
# 43.31605304441928
plt.plot(hw_test,color='red')
plt.plot(delhidata3_akima['pm25'].iloc[2103:])

import pickle
pickle.dump(hw,open('holts_model.pkl','wb'))
