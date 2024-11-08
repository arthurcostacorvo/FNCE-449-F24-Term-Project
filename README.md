# Integrated Asset Selection and Allocation for Portfolio Optimization

This project explores portfolio optimization through an integrated approach that combines asset selection and allocation, 
aiming to construct multiple portfolios with an optimal risk-adjusted return and selecting the top-performing portfolio. 

## Project Structure

The project consists of three main Python files:

1. 01_Data_Handling.py
   - Processes raw financial data, cleans it, and generates CSV files that will be used in subsequent steps.
   - Outputs cleaned data files for easy integration with portfolio optimization scripts.

2. 02_Optimization.py
   - Loads the cleaned CSV files and performs portfolio optimization.
   - Uses the Sharpe ratio as a key metric to identify the optimal combination of stocks from various sectors.
   - Outputs the optimal portfolio weights and other metrics such as expected return and standard deviation.

3. 03_Application.py
   - Accesses the optimal portfolio's weights and stock selection from the previous script.
   - Tracks the portfolio's performance using forward testing, allowing analysis of the portfolio's results over time.

## Getting Started

1. Requirements: 
   - Python 3.x
   - Libraries: `datetime`,`yfinance`, `pandas`, `numpy`, `json`, `scipy`, `tqdm`, `itertools`
   
2. Data Preparation: Run `01_Data_Handling.py` to clean and prepare the data for analysis.

3. Portfolio Optimization: Execute `02_Optimization.py` to identify the optimal portfolio configuration.

4. Tracking Results: Use `03_Application.py` to monitor the performance of the selected portfolio in a forward test environment.

## Notes

- Ensure that the necessary data files: (`backtest_df.csv`, `backtest_rf.csv`,
                                        `frwdtest_df.csv`, `frwdtest_mk.csv`,
                                        `securities_dict.json`, `optimal_portfolio.json`)
  are available in the same directory.
  
