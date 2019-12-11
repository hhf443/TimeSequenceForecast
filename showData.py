import pandas as pd
import matplotlib.pyplot as plt

def showdata(path, date, prediction):
    # Subsetting the dataset
    # Index 11856 marks the end of year 2013
    df = pd.read_csv(path, nrows=43801)
    
    # Creating train and test set
    # Index 10392 marks the end of October 2013
    train = df[0:40000]
    test = df[40001:]
    
    # Aggregating the dataset at daily level
    df['Timestamp'] = pd.to_datetime(df[date], format='%Y/%m/%d %H:%M')
    df.index = df['Timestamp']
    df = df.resample('D').mean()
    
    train['Timestamp'] = pd.to_datetime(train[date], format='%Y/%m/%d %H:%M')
    train.index = train['Timestamp']
    train = train.resample('D').mean()
    
    test['Timestamp'] = pd.to_datetime(test[date], format='%Y/%m/%d %H:%M')
    test.index = test['Timestamp']
    test = test.resample('D').mean()

    #Plotting data
    plt.figure(figsize=(12, 8))
    plt.plot(train.index, train[prediction], label='Train')
    plt.plot(test.index, test[prediction], label='Test')
    plt.legend(loc='best')
    plt.title("Show Row Data")
    plt.show()

    
if __name__ == '__main__':
    showdata('pollution.csv', 'date', 'pm')