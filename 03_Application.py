import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# Recall 'The Market' Ticker
market_ticker = '^GSPC'  # S&P 500

# LOADING DATA
frwdtest_df = pd.read_csv('frwdtest_df.csv', index_col = 0)
frwdtest_mk = pd.read_csv('frwdtest_mk.csv', index_col = 0)
with open('optimal_portfolio.json', 'r') as json_file:
    optimal_portfolio_data = json.load(json_file)

optimal_portfolio = pd.DataFrame.from_dict(optimal_portfolio_data, orient = 'index')
tickers = optimal_portfolio['Stock'].values
weights = optimal_portfolio['Weight'].values

# Creates a copy of the returns dataframe with only the relecant columns (tickers)
filtered_df = frwdtest_df[[ticker for ticker in tickers if ticker in frwdtest_df.columns]].copy()

# DAILY RETURNS
# Calculate the weighted sum (portfolio return) for each row and assign it to a new column
filtered_df['Portfolio Return'] = filtered_df.dot(weights)
filtered_df['Market Return'] = frwdtest_mk['^GSPC'] # Keeps Market Returns the Same

# DAILY BALANCES
# Step 1: Calculate Growth (1 + Portfolio Return) for each row
filtered_df['Portfolio Growth'] = 1 + filtered_df['Portfolio Return']
filtered_df['Market Growth'] = 1 + filtered_df['Market Return']

# Step 2: Calculate the Cumulative Return (Cumulative Product) for each row
filtered_df['Portfolio Balance'] = filtered_df['Portfolio Growth'].cumprod()
filtered_df['Market Balance'] = filtered_df['Market Growth'].cumprod()

# Create the results DataFrame with Returns and Balances
returns_df = filtered_df[['Portfolio Return', 'Portfolio Balance', 'Market Return', 'Market Balance']]
returns_df.index = pd.to_datetime(returns_df.index)

# Plot and save the Portfolio and Market Returns as a PDF
plt.figure(figsize=(12, 6))
plt.plot(returns_df.index, returns_df['Portfolio Return'], color='#4e79a7', label='Portfolio Returns')
plt.plot(returns_df.index, returns_df['Market Return'], color='#e15759', label='Market Returns')
plt.title('Optimized Portfolio vs Market')
plt.xlabel('Date')
plt.ylabel('Return')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Plot and save the Portfolio and Market Balances as a PDF
plt.figure(figsize=(12, 6))
plt.plot(returns_df.index, returns_df['Portfolio Balance'], color='#4e79a7', label='Portfolio Balance')
plt.plot(returns_df.index, returns_df['Market Balance'], color='#e15759', label='Market Balance')
plt.title('$1 Invested Over Time')
plt.xlabel('Date')
plt.ylabel('Balance')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
plt.legend()
plt.xticks(rotation=45)
plt.show()
