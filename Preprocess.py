
# < ---- Import Necessary Packages ---- >
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from termcolor import cprint, colored
import pandas as pd
import numpy as np
from sklearn import preprocessing
import ta  # Technical Analysis library



def Preprocessing(path_file):

    """
    :param path_file: path of the file
    :return: features & labels
    """
    data = pd.read_csv(path_file)

    # convert the date column into a datetime object
    data['Date  '] = pd.to_datetime(data['Date  '])
    # extract the day, month, and year components
    data['day'] = data['Date  '].dt.day
    data['month'] = data['Date  '].dt.month
    data['year'] = data['Date  '].dt.year
    # rename column names
    data.rename(columns={'Close      Close price adjusted for splits.   ': 'Close',
                         'Adj Close      Adjusted close price adjusted for splits and dividend and/or capital gain distributions.   ': 'Adj Close'},
                inplace=True)
    # drop unnecessary columns
    data = data.drop(['Date  ', 'Adj Close'], axis=1)
    # replace nan values
    data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # string to numeric
    data = data.replace('-', 0)
    # remove the separator
    data = data.map(lambda x: x.replace(',', '') if isinstance(x, str) else x)
    data = data.astype(float)
    # Technical Indicators - are added as extra features to get better prediction

    # Simple Moving Average
    data['SMA'] = data['Close'].rolling(window=14).mean()
    # Exponentially-weighted Moving Average
    data['EWMA'] = data['Close'].ewm(span=14, adjust=False).mean()
    # Bollinger Bands
    data['Middle Band'] = data['Close'].rolling(window=20).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=20).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=20).std()
    # Relative Strength Index (RSI)
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
    # Money Flow Index (MFI)
    data['MFI'] = ta.volume.MFIIndicator(high=data['High  '], low=data['Low  '], close=data['Close'], volume=data['Volume  '],
                                    window=14).money_flow_index()
    # Average True Range (ATR)
    data['ATR'] = ta.volatility.AverageTrueRange(high=data['High  '], low=data['Low  '], close=data['Close'],
                                               window=14).average_true_range()
    # Force Index
    data['Force Index'] = data['Close'].diff(1) * data['Volume  ']
    # Ease of Movement
    data['EMV'] = ((data['High  '] + data['Low  ']) / 2 - (data['High  '].shift(1) + data['Low  '].shift(1)) / 2) / (
                data['Volume  '] / (data['High  '] - data['Low  ']))
    # features and labels
    all_data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    feat = all_data.drop(['Close'], axis=1)
    lab = all_data['Close']

    # data Normalization
    feat = preprocessing.normalize(feat)
    print(' --------- ')
    print(feat.shape)
    print(lab.shape)
    print(' --------- ')
    return feat, lab

def linear_regression(xtrain, ytrain, xtest, ytest):
    print(colored("Support Vector Regression  ---->> ", color='blue', on_color='on_grey'))
    model = LinearRegression()
    # model = SVR(kernel='linear')
    # train the data
    model.fit(xtrain, ytrain)
    # predict
    preds = model.predict(xtest)
    return ytest, preds


path = 'BSE_Data.csv'
feat, lab = Preprocessing(path)
# Split feat and lab into train and test
percent = 0.7
counts = feat.shape[0]
sep = int(percent * counts)
# splitting data for training and testing purpose
xtrain, xtest, ytrain, ytest = feat[:sep], feat[sep:], lab[:sep], lab[sep:]
print("xtrain, xtest, ytrain, ytest:", xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)


ytrue, pred = linear_regression(xtrain, ytrain, xtest, ytest)
# Mean Absolute Error
MAE = np.mean(abs(ytrue - pred))
# Mean Squared Error
MSE = mean_squared_error(ytrue, pred)
# Root Mean Squared Error
RMSE = np.sqrt(MSE)

# the prediction scores are not much better 
# to do that packup large amount of data and do the prediction 

