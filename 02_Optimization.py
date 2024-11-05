from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import json

# Recall Security Dictionary Keys
STOCK_1 = 'Stock #1'
STOCK_2 = 'Stock #2'

# LOADING DATA
backtest_df = pd.read_csv('backtest_df.csv', index_col=0)
backtest_rf = pd.read_csv('backtest_rf.csv', index_col=0)
with open('securities_dict.json', 'r') as f:
    securities_dict = json.load(f)

# SETTING RISK-FREE RATE: Last Available Rate
risk_free = backtest_rf['Risk-Free Rate'].iloc[-1]

# Calculates Portfolio Expected Returns
def expected_return(weights, returns_df):
    return np.sum(returns_df.mean() * weights) * 252
    # Annualined Expected Return (252 Trading Days)

# Calculates Portfolio Standard Deviation
def standard_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

# Calculates Portfolio Sharpe Ratio
def sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate):
    portfolio_return = expected_return(weights, returns_df)
    portfolio_stdev = standard_deviation(weights, cov_matrix)
    return (portfolio_return - risk_free_rate) / portfolio_stdev

# Define Negative Sharpe Ratio for Minimization
def neg_sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, returns_df, cov_matrix, risk_free_rate)
# SOURCE: https://www.youtube.com/watch?v=9GA2WlYFeBU&t=973s

# Generates List with All possible Portfolio Combinations
choices = [(sector[STOCK_1], sector[STOCK_2]) for sector in securities_dict.values()]
combinations = list(itertools.product(*choices))

# Uses the first combination to determine constraints and bounds applied to all combinations
first_combination = combinations[1]
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.18) for _ in range(len(first_combination))]
initial_weights = np.array([1 / len(first_combination)] * len(first_combination))

# Main Algo Loop: Calculates Sharpe for each Combination
optimal_sharpe = 0  # Sets initial optimal Sharpe as 0
for combination in tqdm(combinations, desc = 'Evaluating Portfolio Combinations'):
    filtered_df = backtest_df[list(combination)]  # Filter dataframe for stocks in this combination
    covariance_mx = filtered_df.cov() * 252  # Calculate the covariance matrix for the dataframe
    result = minimize(neg_sharpe_ratio, initial_weights, args = (filtered_df, covariance_mx, risk_free),
                      method='SLSQP', constraints = constraints, bounds = bounds)
    # SOURCE: https://www.youtube.com/watch?v=9GA2WlYFeBU&t=973s
 
    sharpe = -result.fun.round(4)

    # Keeps track of the Optimal Sharpe
    if sharpe > optimal_sharpe:
        optimal_sharpe = sharpe                         # Records Optimal Portfolio Sharpe
        optimal_combination = combination               # Records Optimal Combination of Stocks
        optimal_weights = np.round(result.x, 4)         # Records Optimal Portfolio Weights (4 Decimals)
        optimal_returns_df = filtered_df.copy()         # Saves a Copy of Optimal Filtered Dataframe
        optimal_covariance_mx = covariance_mx.copy()    # Saves a Copy of Optimal Covariance Matrix

        
# ORGANIZE RESULTS
data = []
for stock, weight in zip(optimal_combination, optimal_weights):
    # Finds the sector for each stock
    sector = next(sector_name for sector_name, stocks in securities_dict.items() if stock in stocks.values())
    data.append([sector, stock, weight])
# Converts the list to a DataFrame and arrange columns
optimal_portfolio_df = pd.DataFrame(data, columns = ['Sector', 'Stock', 'Weight'])

# Calculates the Optimal Portfolio Return and Standard Deviation
optimal_return = expected_return(optimal_weights, optimal_returns_df).round(4)
optimal_stdev = standard_deviation(optimal_weights, optimal_covariance_mx).round(4)

# PRINT RESULTS
print('\n\tOptimal Portfolio Allocation:\n')
print(optimal_portfolio_df)
print(f'\nPortfolio Return: {optimal_return}')
print(f'Portfolio Standard Deviation: {optimal_stdev}')
print(f'Sharpe Ratio: {optimal_sharpe} \n')

# Save the optimal portfolio to a JSON file
optimal_portfolio_df.to_json('optimal_portfolio.json', orient='index')


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Extract sectors, stocks, and weights from the DataFrame
sectors = optimal_portfolio_df['Sector']
stocks = optimal_portfolio_df['Stock']
weights = optimal_portfolio_df['Weight']

# Create the bar chart using Tableau's blue color
plt.figure(figsize=(12, 8))
bars = plt.bar(sectors, weights, color=mcolors.TABLEAU_COLORS['tab:blue'])

# Add stock names above each bar
for bar, stock in zip(bars, stocks):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), stock, 
             ha='center', va='bottom', fontsize=10)

# Set chart title and axis labels
plt.title('Optimal Portfolio')
plt.xlabel('Sector')
plt.ylabel('Weight')
plt.ylim(0, 0.2)  # Set the bounds from 0 to 0.2

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()

# Display the chart
plt.show()