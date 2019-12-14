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
from statsmodels.tsa.holtwinters import Holt


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

    nullArray = train.copy()
    nullArray['time'] = train.index

    # paramsList = [ col_pm, col_humidity, col_temperature, col_pressure, col_windspeed, col_snowfall, col_rainfall, date]
    #newList = []
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        dd = np.asarray(train[paramsList[i]])
        y_hat[paramsList[i]] = dd[len(dd)-1]
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)


    # --------------------------------------
    y_hat['time']=test.index;

    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)

    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')  #index=False,header=False表示不保存行索引和列标题



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
    nullArray = train.copy()
    nullArray['time'] = train.index


    for i in range(2,len(paramsList)-1):
        y_hat[paramsList[i]] = train[paramsList[i]].mean()
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)
    # --------------------------------------

    y_hat['time'] = test.index

    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""

    s.to_csv(saveto,index=False,header=True)  #index=False,header=False表示不保存行索引和列标题

# 3：移动平均法
def moving_avg_forecast(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date'], specialParams=['60']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]
    windows = specialParams[0]

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
    nullArray = train.copy()
    nullArray['time'] = train.index
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        y_hat[paramsList[i]] = train[paramsList[i]].rolling(int(windows)).mean().iloc[-1]
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)
        # newList.append(paramsList[i])

    # --------------------------------------
    y_hat['time'] = test.index
    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')

    '''
    rms = sqrt(mean_squared_error(test['Count'], y_hat_avg['moving_avg_forecast']))
    print(rms)
    '''
 # 4：简单指数平滑法
def SES(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date'],specialParams=['0.6']):

    '''
    1.时间序列比较平稳时，选择较小的α值，0.05-0.20。

            2.时间序列有波动，但长期趋势没大的变化，可选稍大的α值，0.10-0.40。

            3.时间序列波动很大，长期趋势变化大有明显的上升或下降趋势时，宜选较大的α值，0.60-0.80。

            4.当时间序列是上升或下降序列，满足加性模型，α取较大值，0.60-1。
    '''
    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]
    es = specialParams[0]

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
    nullArray = train.copy()
    nullArray['time'] = train.index
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit = SimpleExpSmoothing(np.asarray(train[paramsList[i]])).fit(smoothing_level=float(es), optimized=False)
        y_hat[paramsList[i]] = fit.forecast(len(test))
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)

        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)

    y_hat['time'] = test.index
    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')

# 5：霍尔特(Holt)线性趋势法
def Holtmethod(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date'], specialParams=['0.3','0.1']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]
    smoothing_level = specialParams[0]
    smoothing_slope = specialParams[1]

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
    nullArray = train.copy()
    nullArray['time'] = train.index
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit = Holt(np.asarray(train[paramsList[i]])).fit(smoothing_level=float(smoothing_level), smoothing_slope=float(smoothing_slope))
        y_hat[paramsList[i]] = fit.forecast(len(test))
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)

        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)

    y_hat['time'] = test.index

    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')

# 6：Holt-Winters季节性预测模型
def Holt_Winters(paramsList=['pollution.csv', '0.93','pm', 'humidity', 'date'], specialParams=['7']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]
    season = specialParams[0]

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
    nullArray = train.copy()
    nullArray['time'] = train.index
    # 以上可通用----------------------------


    for i in range(2,len(paramsList)-1):
        print("进入循环")
        fit1 = ExponentialSmoothing(np.asarray(train[paramsList[i]]), seasonal_periods=int(season), trend='add', seasonal='add').fit()
        y_hat[paramsList[i]] = fit1.predict(start="2014/7/3", end="2014/9/21")
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)
        print("结束fit1")
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)

        y_hat['Holt_Winter'] = fit1.forecast(len(test))
        plt.figure(figsize=(16, 8))
        plt.plot(train[paramsList[i]], label='Train')
        plt.plot(test[paramsList[i]], label='Test')
        plt.plot(y_hat[paramsList[i]], label='Holt_Winter')
        plt.legend(loc='best')
        plt.show()

    y_hat['time'] = test.index
    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')
    '''
    y_hat['Holt_Winter'] = fit1.forecast(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.show()
    '''


# 7：自回归移动平均模型（ARIMA）
def ARIMA(paramsList=['pollution.csv', '0.93','pm','date'], specialParams=['2','1','4','0','1', '1', '7']):

    path = paramsList[0]
    trainRows = float(paramsList[1])
    saveto = 'result.csv'
    df = pd.read_csv(path, usecols=paramsList[2:])
    allRows = df.shape[0]
    order = tuple(map(int, specialParams[0:3].copy()))
    seasonal_order = tuple(map(int, specialParams[3:].copy()))

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
    nullArray = train.copy()
    nullArray['time'] = train.index
    # 以上可通用----------------------------

    for i in range(2,len(paramsList)-1):
        fit1 = sm.tsa.statespace.SARIMAX(train[paramsList[i]], order=order, seasonal_order=seasonal_order).fit()
        # newList.append(paramsList[i])
        #y_hat_avg[paramsList[i]] = fit1.predict(start="2014/7/3", end="2014/12/31", dynamic=True)

        y_hat[paramsList[i]] = fit1.predict(start=test.index[0], end=test.index[-1], dynamic=True)
        y_hat[paramsList[i]] = round(y_hat[paramsList[i]],2)
        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        print(rms)
    # --------------------------------------
    y_hat['time'] = test.index
    yhat_naive = np.array(y_hat)
    nArray = np.array(nullArray)
    newArray = np.concatenate((nArray,yhat_naive),axis=0)
    s = pd.DataFrame(newArray, columns=paramsList[2:])
    for i in range(2,len(paramsList)-1):
        s[paramsList[i]][0:int(len(s)*trainRows)] = ""
    s.to_csv(saveto,index=False,header=True,float_format='%.2f')


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
    Holtmethod()

