"""
Script Name: data.py
Author: Adam Smith-Orlik
Email: asorlik@yorku.ca
Date Created: October 2023
Progress: Development

Description:
Importing and handling stock market data from the yfinance library for use in the backtester package. 
Functionality includes importing, processing, and performing analysis on data for use in building
trading strategies and feature engineering for machine learning purposes. 

Usage:
Instance of the class can be created by running "from data import Data"
To load market data use data = Data(tickers[<ticker symbols>])
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import os
# from utils import *
import requests
# machine learning applications
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# constants available to the class
alpha_vantage_api_key = 'ZKZN82TZL9OHHZWW'

class Data():
    """
    Data class extending the use of yfinance's API to the specific needs of BACKTESTER. Data loads a number of tickers are saves historical data
    for ease of access to the strategy class. Data also provides analysis functions that extend the features of the yf API, including Gaussian
    smoothing kernels, moving average calculators, and a quick view plotting tool. 
    """
    
    def __init__(self, tickers=None, start=datetime.datetime(2021,1,1), end=datetime.datetime.now(), max=False, interval='1d'):
        """
        Args: 
            ticker (str or list of strs) : stock tickers
            start (datetime) : starting date for data acquisition
            end (datetime) : end date for data acquisition
        Returns:
            loads a ticker_data dictionary with key "ticker" and value a yf Ticker object
        """
        # initialize params
        self.tickers = tickers
        self.interval = interval
        # intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
        self.max = max
        if isinstance(tickers, str):
            tickers = [tickers]
        if self.tickers == None:
            raise ValueError("A ticker is required to initialize a data object.")
        self.start = start
        self.end = end
        self.num_tickers = len(tickers)
        # save ticker object
        self.ticker_data = {}
        # save historical data
        self.historical = {}
        
        # save all ticker data in dictionaries to avoid redundant calls
        for ticker in self.tickers:
            # load yf ticker object
            object = yf.Ticker(ticker)
            history = yf.download(ticker, period='max', interval=self.interval) if self.max else yf.download(ticker, start=self.start, end=self.end, interval=self.interval)
            # rename adjusted close 
            history.rename(columns={'Adj Close':'Adj_close'}, inplace=True)
            # add to  dictionaries 
            self.ticker_data[ticker] = object
            self.historical[ticker] = history


    def get_historical(self, ticker, frequency='D'):
        """
        Args: 
            ticker (str) : stock ticker
            frequency (str) : frequency of datetime object. Can be day 'D', month 'M', year 'Y', or quarterly 'Q.'
        Returns:
            historical data resampled to frequency
        """
        df = self.historical[ticker]
        df_sampled = df.resample(frequency).mean()
        return df_sampled

    def get_dates(self, ticker):
        """
        Args: 
            ticker (str) : stock ticker
        Returns:
            list of dates over which data is initiated
        """
        dates = self.historical[ticker].index
        return dates
    
    def get_price_at_date(self, ticker, date=None, feature="Adj_close"):
        """
        Args: 
            ticker (str) : stock ticker
        Kwargs:
            date (DateTime date) : date for which you want feature
            feature (str) : column name in historical dataset 
        Returns:
            int or float for value of feature at specified date
        """
        if date == None:
            raise ValueError("Please enter a date using 'date=<DateTime obj>'")
        
        values = self.historical[ticker][feature]
        value = values[date]
        return value
    
    def get_earnings(self, ticker, period='quarterly'):
        """
        Args: 
            ticker (str) : stock ticker
        Kwargs:
            period ('annual' or 'quarterly') : period of earnings reports available 
        Returns:
            Earnings reports for the specified date range from alpha vantage API
        """
        # alpha vantage api
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={alpha_vantage_api_key}'
        r = requests.get(url)
        d = r.json()

        # Earnings
        earnings = d[period+'Earnings']

        # flip earning to ensure increasing date
        earnings = earnings[::-1]

        select = []
        for index, data in enumerate(earnings):
            date = datetime.datetime.strptime(data['reportedDate'], '%Y-%m-%d')

            if (date >= self.start) and (date <= self.end):
                select.append(index)

        # slice earnings within date range
        earnings = earnings[select[0]:select[-1]]

        df= pd.DataFrame(earnings)
        dates = df['reportedDate'].values
        dates = [datetime.datetime.strptime(date, '%Y-%m-%d' ) for date in dates]
        df['reportedDate'] = dates
        df = df.set_index('reportedDate')
    
        return df

    ####################################################
    ############ FEATURE ENGINEERING FOR ML ############
    ####################################################
    def add_target(self, df, target='pct_daily'):
        """
        NOTE: It is imperative that the TARGET values be shifted back one day, since we are attempting
        to PREDICT the value of the TARGET tomorrow, given data today...
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            target (str) : either percent change daily 'pct_daily' or a binary indicatory for positive '1' or negative '0'
            percent change daily
        Returns:
            Dataframe with added TARGET
        """
        df = df.copy()
        pct = df.Adj_close.pct_change() # period is determined by the resampling

        if target == 'pct_daily':
            df['TARGET'] = pct.shift(-1)
        elif target == 'binary':
            binary_indicator = (pct > 0).astype(int) #np.where(pct > 0, 1, 0) 
            df['pct_daily'] = pct
            df['TARGET'] = binary_indicator.shift(-1)
        elif target == 'dummy_binary':
            df['TARGET'] = self.dummy(df, type='binary')
        elif target == 'dummy_pct':
            df['TARGET'] = self.dummy(df, type='pct')
        else:
            raise ValueError(f"No support for {target}. Must be either 'pct_daily' or 'binary' or 'dummy_binary', or 'dummy_pct'. ")
        return df
    
    def dummy(self, df, type='binary'):
        """
        This function produces a dummy target that is a prescribed combination of the input features. 
        This is used to test the ability of the ML models to accurately pick up on underlying patterns
        in the data. This is similar to the motivation for adding leakage, but it focuses on testing
        multiple features and feature patterns rather than a single feature.
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            None
        Returns:
            Dataframe with added dummy TARGET
        """ 
        df = df.copy()
        mean, std = (0, 0.5)
        noise = np.random.normal(mean, std, len(df))
        dummy =   noise + df.Volume_ratio_day * df.Adj_close_ratio_day * df.Adj_close_ratio_week * np.sin(df.dayofweek) # * np.cos(df.month)
        # NOTE: no need ot shift here since we are trying to predict the dummy target with todays data
        target = (dummy > 0).astype(int) if type=='binary' else dummy

        return  target
    
    def add_diff_features(self, df):
        """
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            None
        Returns:
            Dataframe the ratios of adjusted close and volume over various time horizons
        """
        df = df.copy()
        
        df['Adj_close_first_diff'] = df.Adj_close.diff()
        df['Volume_first_diff'] =  df.Volume.diff()

        df['Adj_close_second_diff'] = df.Adj_close.diff().diff()
        df['Volume_second_diff'] = df.Volume.diff().diff()
    
        return df
    
    def add_ratio_features(self, df):
        """
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            None
        Returns:
            Dataframe the ratios of adjusted close and volume over various time horizons
        """
        df = df.copy()
        day, week, month, quarter, year= (1, 5, 20, 62, 249)
 
        # day
        df['Adj_close_ratio_day'] = df.Adj_close / df.Adj_close.shift(day)
        df['Volume_ratio_day'] = df.Volume / df.Volume.shift(day)
        # week
        df['Adj_close_ratio_week'] = df.Adj_close / df.Adj_close.shift(week)
        df['Volume_ratio_week'] = df.Volume / df.Volume.shift(week)
        # month
        df['Adj_close_ratio_month'] = df.Adj_close / df.Adj_close.shift(month)
        df['Volume_ratio_month'] = df.Volume / df.Volume.shift(month)
        # quarter
        df['Adj_close_ratio_quarter'] = df.Adj_close / df.Adj_close.shift(quarter)
        df['Volume_ratio_quarter'] = df.Volume / df.Volume.shift(quarter)
        # year
        df['Adj_close_ratio_year'] = df.Adj_close / df.Adj_close.shift(year)
        df['Volume_ratio_year'] = df.Volume / df.Volume.shift(year)

        return df
    
    def add_time_features(self, df):
        """
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            None
        Returns:
            Dataframe with added date information 
        """
        df = df.copy()

        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear

        return df

    def add_lag_features(self, df):
        """
        NOTE: Shift are positive because we want to move past data (say 5 days ago) to todays row, indicating what the 
        TARGET was 5 days ago today... 
        NOTE: The TARGET column is already shifted back one day, since we want to predict what the target is tomorrow 
        based on what the data is today...
        NOTE: pct is used for both pct targets and binary targets, since this data is lagged it wont present a bias
        or leakage for either target types. We could instead lag the binary values for binary target, but this is 
        redundant with percent change and therefore does not matter.. t 
        Args:
            df (dataframe) : pandas historical dataframe for ticker 
        Kwargs:
            None
        Returns:
            Dataframe with added lag information 
        """
        df = df.copy()
        pct = df.Adj_close.pct_change() 
        # roughly 249 trading days in each year
        df['pct_lag_week'] = pct.shift(6)
        df['pct_lag_quarter'] = pct.shift(62)
        df['pct_lag_year'] = pct.shift(249)

        df['Adj_lag_week'] = df.Adj_close.shift(6)
        df['Adj_lag_quarter'] = df.Adj_close.shift(62)
        df['Adj_lag_year'] = df.Adj_close.shift(249)

        df['Vol_lag_week'] = df.Volume.shift(6)
        df['Vol_lag_quarter'] = df.Volume.shift(62)
        df['Vol_lag_year'] = df.Volume.shift(249)
        return df
    
    def add_technical_indicators(self, df, ticker):
        df = df.copy()
        
        # moving averages
        df['SMA_10'] = self.moving_average(ticker, window=10)
        df['SMA_30'] = self.moving_average(ticker, window=30)
        df['EMA_10'] = self.exponential_moving_average(ticker, window=10)
        df['EMA_30'] = self.exponential_moving_average(ticker, window=30)
        df['RSI'] = self.RSI(ticker, window=10)
        # MACD
        macd_dict = self.MACD(ticker)
        df['MACD_signal_line'] = macd_dict['signal_line']
        df['MACD_histogram'] = macd_dict['MACD_histogram']
        # volatility
        df['historical_volatility'] = self.historical_volatility(ticker, window=20)
        # TODO: add VIX from chicago board of exchange
        # TODO: add beta of stock 
        # TODO: look into alpha 

        return df
    
    def feature_matrix(self, ticker, target='pct_daily', leak=True, scale=False, dropnan=True, diff_features=True, ratio_features=True, time_features=True, lag_features=True, technical_indicators=True):
        """
        NOTE: Same as build_feature_matrix but includes a column that leaks information so that the model performs very well.
        Used as a validation that the model is able to pick out higher predicting features... 
        Args:
            ticker (str) : stock ticker
        Kwargs:
            target (str) : "pct_daily" for percent change daily or "binary" for a binary classifier 
            leak (bool) : add pct_change data with small error from the future to add "leakage" into the data
            scale (bool) : Flag to use MinMaxScaler to scale the feature column values between 0 and 1
            dropnan (bool) : Flag to return the feature and target dataframes with the nan columns dropped 
            time_features (bool) : Flag to include the time information to the features 
            lag_features (bool) : Flag to include pct lagged 
        Returns:
            A numpy array with the correct 3D dimensions for the LSTM machine learning model
        """
        df = self.historical[ticker].copy()
        
        if leak:
            # Adding "leakage" to the features 
            pct = df.Adj_close.pct_change() 
            # create 10% random error to add to pct so it is not perfectly predictable of the target
            error = np.random.normal(0,pct.std() * 0.1, pct.shape)
            leakage = pct + error
            
            # NOTE: shift(-1) sets todays pct to yesterday, enabling the model to have knowledge of the future.
            # This knowledge constitutes a "leak" in the data and should lead to the model being able to learn
            # how to predict the target from the leaked data. 
            df['leakage'] = leakage.shift(-1).copy()

        # add standalone features 
        if time_features: df = self.add_time_features(df)
        if technical_indicators: df = self.add_technical_indicators(df, ticker)
        if ratio_features: df = self.add_ratio_features(df)
        if diff_features: df = self.add_diff_features(df)
        # add TARGET
        df = self.add_target(df, target=target)
        # add dependent features 
        if lag_features: df = self.add_lag_features(df)

        if dropnan: df = df.copy().dropna()

        features_df = df.drop('TARGET', axis=1)
        target_df = df.TARGET

        if scale:
            # replace infinity values with 0
            features_df.replace([np.inf, -np.inf], 0, inplace=True)
            # scale 
            scaler = MinMaxScaler()
            features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)

        return (features_df, target_df)
    
    
    ###########################################################
    ############ MACHINE LEARNING DATA PREPARATION ############
    ###########################################################

    def pca_reduction(self, features, k=10):

        df = features.copy()

        pca = PCA(n_components=k)
        df_reduced = pca.fit_transform(df)

        cols = [f'col_{i}' for i in range(1, k + 1)]
        idxs = df.index

        df_dressed = pd.DataFrame(df_reduced, columns=cols, index=idxs)

        return df_dressed
    
    def transform_features_LSTM(self, df, period=1):
        """
        Args:
            df (dataframe) : the FEATURES data frame
        Kwargs:
            period (int) : period of sampling for LSTM later
        Returns:
            A numpy array with the correct 3D dimensions for the LSTM machine learning model
        """
        df = df.copy()
        samples, features = df.shape
        npdf = df.to_numpy()
        npdf = npdf.reshape((samples, period, features))

        return npdf

    ###########################################################################
    ############ TRANSFORM DATA FUNCTIONS AND TECHNICAL INDICATORS ############
    ###########################################################################

    def gaussian_kernel(self, size, sigma=1.0):
        """
        Gaussian kernel used to smooth time series data 
        Args:
            s (int or float) : size of Gaussian kernel 
        Kwargs:
            sigma (float) : standard deviation 
        Returns:
            Normalized Gaussian smoothing kernel
        """
        s = np.arange(0, size) - (size - 1.0) / 2.0
        w = np.exp(-s**2 / (2 * sigma**2))
        return w / np.sum(w)
    
    def percent_change(self, ticker, feature="Adj_close", period=1):
        """
        Args:
            ticker (str) : stock ticker 
        Kwargs:
            feature (str) : column name from historical dataset
            period (int) : size of period over which to calculate the percent change 
        Returns:
            Percent change over period of selected feature 
        """
        series = self.historical[ticker][feature]
        return series.pct_change(periods=period)
    
    # simple moving averages
    def moving_average(self, ticker, feature="Adj_close", window=10):
        """
        Args:
            ticker (str) : stock ticker
        Kwargs:
            feature (str) : Column name from historical dataset 
            window (int) : size of rolling average window
        Returns:
            Mean of the rolling average for window size 'window'. Used in momentum based
            trading strategies.
        """
        series = self.historical[ticker][feature]
        return series.rolling(window=window).mean()
    
    # weights recent prices 
    def exponential_moving_average(self, ticker, feature="Adj_close", window=10):
        """
        Args:
            ticker (str) : stock ticker
        Kwargs:
            feature (str) : Column name from historical dataset 
            window (int) : size of rolling EMA span
        Returns:
            Exponential weighted moving average
            
        """
        series = self.historical[ticker][feature]
        return series.ewm(span=window, adjust=False).mean()
    
    def RSI(self, ticker, feature="Adj_close", window=10):
        """
        Relative Strength Index
        Args:
            ticker (str) : stock ticker
        Kwargs:
            feature (str) : Column name from historical dataset 
            window (int) : size of rolling EMA span
        Returns:
            The relative strength index
            
        """
        df = self.historical[ticker].copy()

        df['price_change'] = df[feature].diff()

        # average gains and average losses over the period
        df['Gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
        df['Loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)

        avg_gain = df['Gain'].rolling(window=window).mean()
        avg_loss = df['Loss'].rolling(window=window).mean()

        relative_strength = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + relative_strength))

        return rsi
    
    def MACD(self, ticker, short_window=12, long_window=26, signal_window=9):
        """
        MACD indicator 
        Args:
            ticker (str) : stock ticker
        Kwargs: 
            short_window (int) : short window for EMA
            long_window (int) : long window for EMA
            signal_window (int) : window for signal EMA
        Returns:
            The signal line and MACD histogram for the ticker provided as a dictionary 
        """
        # short ema
        ema_short = self.exponential_moving_average(ticker, window=short_window)

        # long ema
        ema_long = self.exponential_moving_average(ticker, window=long_window)

        # MACD line (Short-term EMA - Long-term EMA)
        macd = ema_short - ema_long

        # signal line
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()

        # Calculate the MACD Histogram (MACD - Signal)
        macd_histogram = macd - signal_line

        return {"signal_line":signal_line, "MACD_histogram": macd_histogram}
    
    def bollinger_bands(self, ticker, window=20, multiplyer=2):
        """ 
        Args:
            ticker (str) : stock ticker
        Kwargs: 
            window (int) : short window for defining median band
            multiplyer (int or float) : multiplyer on the std from the median
        Returns:
            Dictionary with median, upper, lower bands and the window and multiplyer used to generate them.
        """
        df = self.historical[ticker].copy()

        # middle band
        median = df.Adj_close.rolling(window=window).mean()
        # standard deviation from mean
        std = df.Adj_close.rolling(window=window).std()
        # upper band
        upper = median + (multiplyer * std)
        # lower band 
        lower = median - (multiplyer * std)

        return {"median":median, "upper":upper, "lower":lower, "window":window, "multiplyer":multiplyer}
    
    ##############################################
    ############ PERFORMANCE MEASURES ############
    ##############################################
    def historical_volatility(self, ticker, window=20):
        """
        Args:
            ticker (str) : stock ticker
        Kwargs: 
            window (int) : volatility window
        Returns:
            Historical volatility calculated as sqrt(window) * STD(percent daily change in window) 
        """

        df = self.historical[ticker].copy()

        pct = df.Adj_close.pct_change()

        hv = np.sqrt(window) * pct.rolling(window=window).std() 

        return hv

    def beta(self, ticker):

        # load stock data
        df = self.historical[ticker].copy()
        start, end = df.index[0], df.index[-1]

        # load index
        try:
            # if index data exists up to the same date as the input stock
            index = yf.download("SPX", start=start, end=end)
        except:
            # otherwise use the index maximum to define the length of the input stock 
            index = yf.download("SPX", period="max")
            df = df.copy()[-len(index):]
    
        pct_stock = df.Adj_close.pct_change().dropna()
        pct_index = df.Adj_close.pct_change().dropna()

        covariance = np.cov(pct_stock, pct_index)[0,1]

        index_variance = np.var(pct_index)

        beta = covariance / index_variance

        return beta
    
    def alpha(self, ticker):

        # load stock data
        df = self.historical[ticker].copy()
        start, end = df.index[0], df.index[-1]

        # load index
        try:
            # if index data exists up to the same date as the input stock
            index = yf.download("SPX", start=start, end=end)
        except:
            # otherwise use the index maximum to define the length of the input stock 
            index = yf.download("SPX", period="max")
            df = df.copy()[-len(index):]

        pct_stock = df.Adj_close.pct_change().dropna()
        pct_index = df.Adj_close.pct_change().dropna()

        risk_free_rate = 0.01 * np.ones(len(pct_index))

        beta = self.beta(ticker)

        expected_returns = risk_free_rate + beta * (np.mean(pct_index) - risk_free_rate)
        returns = np.mean(pct_stock)

        alpha = returns - expected_returns

        return alpha


    @classmethod
    def info(cls):
        """List all methods and the associated docstrings for the class."""
        for attr_name in dir(cls):
            if not attr_name.startswith("__"):
                docstring = getattr(cls, attr_name).__doc__
                print(f'Method Name: {attr_name}\nDocstring: {docstring}\n')

    ############ VISUALIZING ############
    # plot each time series data for the loaded tickers
    def quick_view(self, feature="Adj_close", resolution=1, save=False, logy=False):
        """""
        Plot the input list of tickers for a quick view of the data loaded. 

        Args: 
            None
        Kwargs: 
            resolution (int) : sets the resolution of the plotted data, with the minimum being 1 day resolution 
        Returns:
            Time series plot over the specified range including 10 and 30 day moving averages
        """""
        data = []
        for k,v in self.historical.items():
            d = self.historical[k][::resolution]
            data.append(d)
        data = pd.concat(data, keys=self.historical.keys(), names=['Ticker'])

        # create plot of feature, percent change and volume
        for i in range(self.num_tickers):
            ticker = self.tickers[i]
            # print(ticker)
            d = data.loc[ticker]
            # print(d[feature])
            dates = d.index.date
            # moving averages
            mavg_10 = self.moving_average(ticker, window=10)
            mavg_30 = self.moving_average(ticker, window=30)
            # percent change 
            mid_point = 1/2 * ( d[feature].max() + d[feature].min() )
            y_range = d[feature].max() - d[feature].min()
            pct = self.percent_change(ticker).dropna() * (y_range * 0.8) # 80% of the plot range
            percent_change = pct + mid_point # scale pct to midpoint of plot
            # Compute histogram values without plotting
            hist_vals, edges = np.histogram(pct, bins=30)
            bin_heights = edges[1:] - edges[:-1]
            # resample volume to fill skipped days
            volume = d.Volume.resample('D').ffill()
            # scale volume
            order_of_magnitude = np.floor(np.log10(max(np.abs(volume))))
            scale_factor = 10**order_of_magnitude
            volume_scaled = volume / scale_factor

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6), gridspec_kw={'width_ratios': [6, 1], 'height_ratios': [3, 1]})

            # main plot
            axes[0,0].plot(d[feature],  color='k', ls='-', lw=1, label=feature)
            axes[0,0].plot(mavg_10, color='C0', ls='--', lw=1, alpha = 0.8, label='10 day MAVG')
            axes[0,0].plot(mavg_30, color='C1', ls='--', lw=1, alpha = 0.8, label='30 day MAVG')
            axes[0,0].plot(percent_change, color='k', alpha=0.3, lw=0.5, zorder=1, label="PCT")
            axes[0,0].set(ylabel="Price", title=ticker) 
            axes[0,0].set_xticks([])  
            if logy: axes[0,0].set_yscale('log')
            # axes[0,0].set_xticklabels(dates, rotation=45, fontsize=8)
            axes[0,0].legend(loc='best')

            # histogram of pct
            axes[0,1].barh(edges[:-1], hist_vals, height=bin_heights, align='edge',color='lightgray', label='Daily PCT')
            axes[0,1].axhline(0, ls='--', alpha=0.6, color='k')
            axes[0,1].invert_yaxis()
            axes[0,1].yaxis.tick_right()  
            axes[0,1].set_xticks([])
            axes[0,1].legend(loc='best')

            # volume plot
            axes[1,0].bar(volume.index, volume_scaled.values, edgecolor='lightgray')
            axes[1,0].set_xlabel("Date")
            axes[1,0].set_ylabel(r"Volume [$\times 10^{%d}$]" %order_of_magnitude)

            # black right corner plot
            axes[1,1].axis('off')

            # add statistics to the blank area 
            mean, std, minp, maxp = np.mean(pct), np.std(pct), np.min(pct), np.max(pct)

            stats = """
            Statistics: \nMean: %.3f \nStd: %.3f \nMin: %.3f \nMax: %.3f
            """ %(mean, std, minp, maxp)
            
            axes[1, 1].text(0.5, 0.5, stats.strip(), transform=axes[1, 1].transAxes,
                ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='white'))


            fig.subplots_adjust(wspace=0)
            fig.subplots_adjust(hspace=0)

            if save:
                path = os.getcwd()
                parent = os.path.dirname(path)
                data_path = os.path.join(parent,"data/")
                plt.savefig(data_path + ticker + "_" + dates[0].strftime('%Y_%m_%d') + "_to_" + dates[-1].strftime('%Y_%m_%d') + "_quickview.pdf")

            plt.show()
            plt.clf()