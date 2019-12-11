import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
import re

#def naive(path, date='date', col_pm='pm', col_humidity='humidity', col_temperature='temperature',
#          col_pressure='pressure', col_windspeed='windspeed', col_snowfall='snowfall', col_rainfall='rainfall'):
def naive(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):
    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    #saveto='naive_'+path[:-4]+'.csv'
    # 1：朴素法
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    # Creating train and test set
    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    # Aggregating the dataset at daily level 每天为单位聚合数据集
    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    #timeArray = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # paramsList = [ col_pm, col_humidity, col_temperature, col_pressure, col_windspeed, col_snowfall, col_rainfall, date]
    #newList = []
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        dd = np.asarray(train[paramsList[i]])
        y_hat[paramsList[i]] = dd[len(dd)-1]
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)


    # --------------------------------------
    y_hat['time']=test.index;
    yhat_naive = np.array(y_hat)

    s = pd.DataFrame(yhat_naive, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)  #index=False,header=False表示不保存行索引和列标题



def avg_forecast(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):
    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    #saveto='naive_'+path[:-4]+'.csv'
    # 2：简单平均法
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    # Creating train and test set
    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    # Aggregating the dataset at daily level 每天为单位聚合数据集
    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    #timeArray = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # paramsList = [ col_pm, col_humidity, col_temperature, col_pressure, col_windspeed, col_snowfall, col_rainfall, date]
    #newList = []
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        y_hat[paramsList[i]] = train[paramsList[i]].mean()
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)
    # --------------------------------------

    y_hat['time'] = test.index
    yhat_avg = np.array(y_hat)
    s = pd.DataFrame(yhat_avg, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)  #index=False,header=False表示不保存行索引和列标题


# 3：移动平均法
def moving_avg_forecast(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        y_hat[paramsList[i]] = train[paramsList[i]].rolling(60).mean().iloc[-1]
        # newList.append(paramsList[i])

    # --------------------------------------
    y_hat['time'] = test.index
    yhat_avg = np.array(y_hat)
    s = pd.DataFrame(yhat_avg, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)

    '''
    rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['moving_avg_forecast']))
    print(rms)
    '''
 # 4：简单指数平滑法
def SES(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit = SimpleExpSmoothing(np.asarray(train[paramsList[i]])).fit(smoothing_level=0.6, optimized=False)
        y_hat[paramsList[i]] = fit.forecast(len(test))

        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)

    y_hat['time'] = test.index
    yhat_avg = np.array(y_hat)
    s = pd.DataFrame(yhat_avg, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)

# 5：霍尔特(Holt)线性趋势法
def Holt(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit = Holt(np.asarray(train[paramsList[i]])).fit(smoothing_level=0.3, smoothing_slope=0.1)
        y_hat[paramsList[i]] = fit.forecast(len(test))

        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)

    y_hat['time'] = test.index
    yhat_avg = np.array(y_hat)
    s = pd.DataFrame(yhat_avg, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)

# 6：Holt-Winters季节性预测模型
def Holt_Winters(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # 以上可通用----------------------------


    for i in range(2,len(paramsList)-1):
        print(1)
    fit1 = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    y_hat['Holt_Winter'] = fit1.forecast(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.show()

    rms = sqrt(mean_squared_error(test['Count'], y_hat['Holt_Winter']))
    print(rms)

# 7：自回归移动平均模型（ARIMA）
def ARIMA(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]

    train = df[0:int(allRows*trainRows)]
    test = df[int(allRows*trainRows)+1:]

    print(test["Timestamp"])
    df['Timestamp'] = pd.to_datetime(df[paramsList[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()

    train['Timestamp'] = pd.to_datetime(train[paramsList[-1]], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()

    test['Timestamp'] = pd.to_datetime(test[paramsList[-1]], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    y_hat = test.copy()
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit1 = sm.tsa.statespace.SARIMAX(train[paramsList[i]], order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
        # newList.append(paramsList[i])
        #y_hat_avg[paramsList[i]] = fit1.predict(start="2014/7/3", end="2014/12/31", dynamic=True)

        y_hat[paramsList[i]] = fit1.predict(start="2014/7/3", end="2014/12/31", dynamic=True)
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)
    # --------------------------------------
    y_hat['time'] = test.index
    yhat_avg = np.array(y_hat)
    s = pd.DataFrame(yhat_avg, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)


if __name__ == '__main__':

    '''
        list = [ 'pm','humidity', 'date']
        #list = ['Count', 'Datetime']
        naive(path='pollution.csv', trainRows=0.93, paramsList=list, saveto='naive_new.csv')
        #naive(path='train.csv', trainRows=0.93, paramsList=list, saveto='naive_train.csv')

        #avg_forecast(path='pollution.csv',  trainRows=0.90, paramsList=list,saveto='avg_train.csv')
        #moving_avg_forecast(path='pollution.csv', trainRows=0.90, paramsList=list)
        #SES(path='pollution.csv',  trainRows=0.90, paramsList=list)
        #Holt(path='pollution.csv', trainRows=0.90, paramsList=list)
        # Holt_Winters('train.csv')
        #ARIMA(path='train.csv', allRows=18280, trainRows=0.90, paramsList=list)
        #ARIMA(path='pollution.csv', trainRows=0.90, paramsList=list)
    '''
    ARIMA()

