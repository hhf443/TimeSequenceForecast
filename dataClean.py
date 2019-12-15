import pandas as pd
import numpy as np

def dataclean(params=['pollution.csv','date']):
    df = pd.read_csv(params[0])
    list = df.columns.tolist()
    df['Timestamp'] = pd.to_datetime(df[params[-1]], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()


    for i in range(1,len(list)-1):
        df[list[i]] = round(df[list[i]], 2)
    df['time'] = df.index
    array = np.array(df)

    temp = list[0]
    list[0] = list[-1]
    list[-1] = temp

    s = pd.DataFrame(array, columns=list)
    s.to_csv(params[0],index=False,header=True)


if __name__ == '__main__':
    dataclean()
