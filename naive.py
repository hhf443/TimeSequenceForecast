import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

import sys
from sklearn.metrics import mean_squared_error

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

        # newList.append(paramsList[i])
        plt.figure(figsize=(12, 8))
        plt.plot(train.index, train[paramsList[i]], label='Train')
        plt.plot(test.index, test[paramsList[i]], label='Test')
        plt.plot(y_hat.index, y_hat[paramsList[i]], label='Naive Forecast')
        # data_write_csv('naiveFile.csv')
        plt.legend(loc='best')
        plt.title("Naive Forecast")
        #plt.show()

        rms = sqrt(mean_squared_error(test[paramsList[i]], y_hat[paramsList[i]]))
        #print(rms)


    # --------------------------------------
    y_hat['time']=test.index;
    yhat_naive = np.array(y_hat)

    s = pd.DataFrame(yhat_naive, columns=paramsList[2:])
    s.to_csv(saveto,index=False,header=True)  #index=False,header=False表示不保存行索引和列标题

if __name__ == '__main__':

    #naive(path='pollution.csv', trainRows=0.93, paramsList=list, saveto='naive_new.csv')
    #naive(path=a[0], trainRows=a[1], paramsList=a[2], saveto='result_naive.csv')
    naive()
    print("预测方法调用完成")
