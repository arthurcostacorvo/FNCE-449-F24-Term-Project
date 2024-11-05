from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json

# Date Range
time_range = 365  # Time range for Backtest, Forward Test; in Days
end_date = datetime(2024, 11, 3)
beg_date = end_date - timedelta(days = 2 * time_range)

# Define Security Dictionary Keys
STOCK_1 = 'Stock #1'
STOCK_2 = 'Stock #2'

# Define 'The Market' and 'The Risk-Free' Tickers
market_ticker    = '^GSPC'  # S&P 500
risk_free_ticker = '^TNX'   # CBOE Interest Rate 10 Year T Note

# Define the Security Dictionary with Stock Tickers
securities_dict = {
    'Information Technology':   {STOCK_1: 'AAPL',  STOCK_2: 'MSFT'},
    'Healthcare':               {STOCK_1: 'LLY',   STOCK_2: 'NVO'},
    'Financials':               {STOCK_1: 'JPM',   STOCK_2: 'V'},
    'Consumer Discretionary':   {STOCK_1: 'AMZN',  STOCK_2: 'TSLA'},
    'Consumer Staples':         {STOCK_1: 'WMT',   STOCK_2: 'COST'},
    'Energy':                   {STOCK_1: 'BP',    STOCK_2: 'EQNR'},
    'Industrials':              {STOCK_1: 'GE',    STOCK_2: 'CAT'},
    'Materials':                {STOCK_1: 'LIN',   STOCK_2: 'BHP'},
    'Utilities':                {STOCK_1: 'NEE',   STOCK_2: 'SO'},
    'Real Estate':              {STOCK_1: 'PLD',   STOCK_2: 'AMT'},
    'Communication Services':   {STOCK_1: 'GOOGL', STOCK_2: 'META'}}

# Converts Dictionary into a List of Stocks
def get_securities(dictionary):
    securities_lst = []
    for sector in dictionary.values():
        securities_lst.extend([sector[STOCK_1], sector[STOCK_2]])
    return securities_lst

# Retrieve Adjusted Close Prices for All Securities
def get_data(tickers, beg_date, end_date):
    dataframe = pd.DataFrame()
    # If statements to approach Market Returns (str), and Securities (dict)
    if isinstance(tickers, str):
        data = yf.download(tickers, start=beg_date, end=end_date)
        dataframe[tickers] = data['Adj Close']
        dataframe.index = dataframe.index.strftime('%Y-%m-%d')  # Format dates to 'yyyy-mm-dd'
        return dataframe

    elif isinstance(tickers, list):
        for ticker in tickers:
            data = yf.download(ticker, start=beg_date, end=end_date)
            dataframe[ticker] = data['Adj Close']
        dataframe.index = dataframe.index.strftime('%Y-%m-%d')  # Format dates to 'yyyy-mm-dd'
        return dataframe

    else:
        raise ValueError('Tickers must be a string or a list.')

# Calculate Daily Log Returns for each Stock
def calc_returns(dataframe):
    lnreturn_df = np.log(dataframe / dataframe.shift(1))
    lnreturn_df.dropna(inplace = True)
    lnreturn_df = lnreturn_df.round(4)  # Round to 4 decimal places
    return lnreturn_df

# Splits the Dataframe for Back and Forward Tests
def split_data(dataframe):
    midpoint = len(dataframe) // 2
    backtest_df = dataframe.iloc[:midpoint]
    frwdtest_df = dataframe.iloc[midpoint:]
    return backtest_df, frwdtest_df

# Retrieve and format the risk-free rate data
def get_risk_free(ticker, beg_date, end_date):
    dataframe = pd.DataFrame()
    data = yf.download(ticker, start=beg_date, end=end_date)
    dataframe['Risk-Free Rate'] = (data['Adj Close'] / 100).round(4)  # Convert to decimal and round to 4 decimals
    dataframe.index = dataframe.index.strftime('%Y-%m-%d')  # Format dates to 'yyyy-mm-dd'
    return dataframe

# EXECUTION
## Security Returns
securities_lst = get_securities(securities_dict)
adjclose_df = get_data(securities_lst, beg_date, end_date)
lnreturn_df = calc_returns(adjclose_df)
backtest_df, frwdtest_df = split_data(lnreturn_df)

## Market Reuturns 
adjclose_mkt = get_data(market_ticker, beg_date, end_date)
lnreturn_mkt = calc_returns(adjclose_mkt)
backtest_mk, frwdtest_mk = split_data(lnreturn_mkt)

## Risk-Free Returns
adjclose_rf = get_risk_free(risk_free_ticker, beg_date, end_date)
backtest_rf, frwdtest_rf = split_data(adjclose_rf)

# Exporting results to CSV
backtest_df.to_csv('backtest_df.csv')
backtest_rf.to_csv('backtest_rf.csv')
frwdtest_df.to_csv('frwdtest_df.csv')
frwdtest_mk.to_csv('frwdtest_mk.csv')

# Save securities_dict to a JSON file
with open('securities_dict.json', 'w') as f:
    json.dump(securities_dict, f)