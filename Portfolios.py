import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from scipy.optimize import minimize


class Portfolios:
    def __init__(self, iter_record: List[Dict],
                price_dict: Dict, risk_free_rate: float = 0.02,
                index_data=None):
        self.iter_record = iter_record
        self._extract_price(price_dict)
        self.index_data = index_data
        self.predictions: Optional[Dict] = self._extract_predictions()
        self.timeseries: Optional[List] = self._extract_timeseries()
        self.risk_free_rate = risk_free_rate
        self.weight_sets = {}
        self.return_sets = {}
        self.return_df = None
    
    def _extract_predictions(self):
        """
        Extract predictions from iteration records.

        This method processes the iteration records provided during the initialization of the Portfolios class.
        It extracts the prediction results for each timestamp and stores them in a dictionary.

        Returns:
            predictions (dict): A dictionary where the keys are timestamps and the values are DataFrames
                                containing the prediction results for each timestamp.
        """
        predictions = {}
        for record in self.iter_record:
            timestamp = record["timestamp"]  # Extract the timestamp from the record
            dataframe = record["results"]    # Extract the prediction results DataFrame from the record
            predictions[timestamp] = dataframe  # Store the DataFrame in the dictionary with the timestamp as the key

        return predictions  # Return the dictionary containing all the predictions

    def _extract_timeseries(self):
        """
        Extract and sort timestamps from iteration records.

        This method processes the iteration records provided during the initialization of the Portfolios class.
        It extracts the timestamps from each record, sorts them in ascending order, and stores the earliest timestamp.

        Returns:
            timeseries (list): A sorted list of timestamps extracted from the iteration records.
        """
        timeseries = sorted([record['timestamp'] for record in self.iter_record], reverse=False)  # Extract and sort timestamps
        self.timeseries_start = timeseries[0]  # Store the earliest timestamp
        return timeseries  # Return the sorted list of timestamps

    def _extract_price(self, price_dict):
        """
        Extract and format price data.

        This method processes the price data provided during the initialization of the Portfolios class.
        It extracts specific columns from the price data for each stock and stores them in a dictionary.

        Args:
            price_dict (dict): A dictionary where the keys are stock symbols and the values are DataFrames
                               containing the price data for each stock.

        Returns:
            None
        """
        empty_dict = {}
        self.timeseries = self._extract_timeseries()  # Extract and store the timeseries
        for key, value in price_dict.items():
            df = value.loc[:, ['open', 'high', 'low', 'close', 'ADJCLOSE', 'DAYRET']]  # Extract specific columns
            empty_dict[key] = df  # Store the formatted DataFrame in the dictionary
        self.price_dict = empty_dict  # Update the class attribute with the formatted price data

    def _vol_creator(self, pool: List[str], ret_col: str = 'DAYRET', vol_col: str = 'vol', window: int = 60):
        """
        Create simple rolling volatility for the given pool of stocks.

        This method calculates the rolling standard deviation (volatility) of the specified return column
        for each stock in the provided pool. The calculated volatility is stored in a new column in the price data.

        Args:
            pool (List[str]): A list of stock symbols for which to calculate the volatility.
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            vol_col (str): The name of the column to store the calculated volatility. Default is 'vol'.
            window (int): The rolling window size for calculating the standard deviation. Default is 60.

        Returns:
            None
        """
        for key in pool:
            value = self.price_dict[key].copy(deep=True)  # Create a deep copy of the price data for the stock
            value[vol_col] = value[ret_col].rolling(window=window).std()  # Calculate rolling standard deviation
            self.price_dict[key]

    def  _ewm_vol_creator(self, pool: List[str], ret_col: str = 'DAYRET',
                         vol_col: str = 'emw_vol',
                         param_type: str = 'com', ewm_param: Optional[float] = None, window: int = 60):
        """
        Create exponentially weighted moving volatility for the given pool of stocks.

        This method calculates the exponentially weighted moving standard deviation (volatility) of the specified return column
        for each stock in the provided pool over a rolling window. The calculated volatility is stored in a new column in the price data.

        Args:
            pool (List[str]): A list of stock symbols for which to calculate the volatility.
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            vol_col (str): The name of the column to store the calculated volatility. Default is 'emw_vol'.
            param_type (str): The type of parameter for the EWM calculation ('com', 'span', 'halflife', 'alpha').
            ewm_param (Optional[float]): The parameter value for the EWM calculation.
            window (int): The rolling window size for applying the EWM calculation. Default is 60.

        Returns:
            None
        """
        for key in pool:
            value = self.price_dict[key].copy(deep=True)  # Create a deep copy of the price data for the stock
            if param_type == 'com':
                value[vol_col] = value[ret_col].rolling(window=window).apply(lambda x: x.ewm(com=ewm_param).std().iloc[-1])
            elif param_type == 'span':
                value[vol_col] = value[ret_col].rolling(window=window).apply(lambda x: x.ewm(span=ewm_param).std().iloc[-1])
            elif param_type == 'halflife':
                value[vol_col] = value[ret_col].rolling(window=window).apply(lambda x: x.ewm(halflife=ewm_param).std().iloc[-1])
            elif param_type == 'alpha':
                value[vol_col] = value[ret_col].rolling(window=window).apply(lambda x: x.ewm(alpha=ewm_param).std().iloc[-1])
            else:
                raise ValueError(f"Invalid param_type: {param_type}")

            self.price_dict[key] = value  # Update the price data with the new volatility column

    def _pool_selection(self, timestamp: pd.Timestamp, pool_size: int = 50):
        """
        Select a pool of stocks based on predictions at a given timestamp.

        This method selects the top stocks based on their prediction scores at a specified timestamp.
        The selected stocks are limited to a specified pool size.

        Args:
            timestamp (pd.Timestamp): The timestamp at which to select the stocks.
            pool_size (int): The number of top stocks to select. Default is 50.

        Returns:
            selected_stocks (list): A list of selected stock symbols.
            prediction (pd.DataFrame): The DataFrame containing the sorted predictions for the selected stocks.
        """
        prediction = self.predictions[timestamp]  # Get the prediction DataFrame for the given timestamp
        prediction = prediction.sort_values(by='Prediction', ascending=False).head(pool_size)  # Sort and select top stocks
        selected_stocks = prediction['Symbol'].tolist()  # Extract the list of selected stock symbols
        return selected_stocks, prediction  # Return the selected stocks and the sorted prediction DataFrame

    def _markowitz_df(self, df: pd.DataFrame, stock: str,
                      end_time: pd.Timestamp, window: int = 50):
        """
        Prepare data for Markowitz portfolio optimization.

        This method extracts a window of return data for a given stock up to a specified end time.
        The extracted data is used for calculating the covariance matrix in Markowitz portfolio optimization.

        Args:
            df (pd.DataFrame): The DataFrame containing the price data for the stock.
            stock (str): The stock symbol.
            end_time (pd.Timestamp): The end time up to which the data is extracted.
            window (int): The rolling window size for extracting the data. Default is 50.

        Returns:
            dayret (pd.Series): A Series containing the return data for the specified stock, renamed with the stock symbol.
        """
        end_pos = df.index.get_loc(end_time)  # Get the position of the end_time in the DataFrame index
        start_pos = end_pos - window  # Calculate the start position based on the window size
        df = df.iloc[start_pos:end_pos + 1]  # Extract the data within the window
        dayret = df['DAYRET'].rename(stock)  # Extract the 'DAYRET' column and rename it with the stock symbol
        return dayret  # Return the extracted return data as a Series

    def _sharpe_ratio(self, weights, predicted_returns, cov_matrix):
        """
        Calculate the Sharpe ratio for a given set of weights, predicted returns, and covariance matrix.

        This method calculates the Sharpe ratio, which is a measure of risk-adjusted return, for a portfolio.
        The Sharpe ratio is calculated as the difference between the portfolio return and the risk-free rate,
        divided by the portfolio volatility. The result is negated because the optimization process typically
        involves minimizing the objective function.

        Args:
            weights (np.ndarray): The weights of the assets in the portfolio.
            predicted_returns (np.ndarray): The predicted returns of the assets in the portfolio.
            cov_matrix (np.ndarray): The covariance matrix of the asset returns.

        Returns:
            float: The negative Sharpe ratio of the portfolio.
        """
        # Calculate the portfolio return
        portfolio_return = np.dot(weights, predicted_returns)

        # Calculate the portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Calculate the Sharpe ratio (negative because we are minimizing)
        return -(portfolio_return - self.risk_free_rate) / portfolio_volatility

    def vol_timing(self, ret_col='DAYRET', vol_type='simple', port_name: str = 'vol_timing',
                   pool_size: int = 50, vol_col='vol',
                   ewm_param=5, window=None):
        """
        Implement a volatility timing strategy for portfolio construction.

        This method constructs a portfolio based on a volatility timing strategy. It calculates the volatility
        for a pool of stocks and assigns weights inversely proportional to the squared volatility.

        Args:
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            vol_type (str): The type of volatility calculation ('simple' or one of the EWM types: 'com', 'span', 'halflife', 'alpha').
            port_name (str): The name of the portfolio. Default is 'vol_timing'.
            pool_size (int): The number of top stocks to select for the pool. Default is 50.
            vol_col (str): The name of the column to store the calculated volatility. Default is 'vol'.
            ewm_param (int): The parameter value for the EWM calculation. Default is 5.
            window (int): The rolling window size for calculating the volatility. Default is None.

        Returns:
            None
        """
        weight_set = {}

        for time in self.timeseries:
            # Select the pool of stocks based on predictions at the given timestamp
            pool, _ = self._pool_selection(time, pool_size)

            # Calculate volatility based on the specified type
            if vol_type == 'simple':
                self._vol_creator(window=window, pool=pool, vol_col=vol_col)
            elif vol_type in ['com', 'span', 'halflife', 'alpha']:
                self._ewm_vol_creator(param_type=vol_type, ewm_param=ewm_param, window=window,
                                      pool=pool, ret_col=ret_col, vol_col=vol_col)
            else:
                raise ValueError(f"Invalid vol_type: {vol_type}")

            # Create a dictionary of volatilities for the selected pool
            pool_vol_dict = {key: self.price_dict[key] for key in pool}
            vols = {key: df.loc[time, vol_col] for key, df in pool_vol_dict.items()}

            # Calculate inverse squared volatilities
            inv_sq_vols = {key: 1 / vol ** 2 for key, vol in vols.items()}
            total_inv_vol = sum(inv_sq_vols.values())

            # Assign weights inversely proportional to the squared volatilities
            weights = {key: inv_vol / total_inv_vol for key, inv_vol in inv_sq_vols.items()}
            weight_set[time] = weights

        # Update the weight sets with the new portfolio
        self.weight_sets.update({port_name: weight_set})


    def rtr_timing(self, ret_col='DAYRET', vol_type='simple', port_name: str = 'rtr_timing',
                   ewm_param=5, pool_size: int = 50, vol_col='vol', window=None):
        """
        Implement a return-to-risk timing strategy for portfolio construction.

        This method constructs a portfolio based on a return-to-risk timing strategy. It calculates the return-to-risk ratio
        for a pool of stocks and assigns weights proportional to the return-to-risk ratio.

        Args:
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            vol_type (str): The type of volatility calculation ('simple' or one of the EWM types: 'com', 'span', 'halflife', 'alpha').
            port_name (str): The name of the portfolio. Default is 'rtr_timing'.
            ewm_param (int): The parameter value for the EWM calculation. Default is 5.
            pool_size (int): The number of top stocks to select for the pool. Default is 50.
            vol_col (str): The name of the column to store the calculated volatility. Default is 'vol'.
            window (int): The rolling window size for calculating the volatility. Default is None.

        Returns:
            None
        """
        weight_set = {}

        for time in self.timeseries:
            # Select the pool of stocks based on predictions at the given timestamp
            pool, prediction = self._pool_selection(time, pool_size)
            pred_dict = prediction.set_index('Symbol')['Prediction'].to_dict()

            # Calculate volatility based on the specified type
            if vol_type == 'simple':
                self._vol_creator(window=window, pool=pool, vol_col=vol_col)
            elif vol_type in ['com', 'span', 'halflife', 'alpha']:
                self._ewm_vol_creator(param_type=vol_type, ewm_param=ewm_param, window=window,
                                      pool=pool, ret_col=ret_col, vol_col=vol_col)
            else:
                raise ValueError(f"Invalid vol_type: {vol_type}")

            # Create a dictionary of volatilities for the selected pool
            pool_vol_dict = {key: self.price_dict[key] for key in pool}
            vols = {key: df.loc[time, vol_col] for key, df in pool_vol_dict.items()}
            inv_sq_vols = {key: 1 / vol ** 2 for key, vol in vols.items()}

            # Calculate return-to-risk ratio for each stock
            rtr_dict = {}
            for code in pool:
                pred_ret = pred_dict[code]
                inv_sq_vol = inv_sq_vols[code]
                rtr = pred_ret / inv_sq_vol
                rtr_dict[code] = rtr

            # Calculate weights proportional to the return-to-risk ratio
            total_rtr = sum(rtr_dict.values())
            weights = {key: rtr / total_rtr for key, rtr in rtr_dict.items()}
            weight_set[time] = weights

        # Update the weight sets with the new portfolio
        self.weight_sets.update({port_name: weight_set})

    def markowitz(self, ret_col='DAYRET', port_name: str = 'markowitz',
                  pool_size: int = 50, window: int = 60,
                  shrinkage=False):
        """
        Implement Markowitz portfolio optimization.

        This method constructs a portfolio based on Markowitz mean-variance optimization. It selects a pool of stocks,
        calculates the covariance matrix of their returns, and optimizes the portfolio weights to maximize the Sharpe ratio.

        Args:
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            port_name (str): The name of the portfolio. Default is 'markowitz'.
            pool_size (int): The number of top stocks to select for the pool. Default is 50.
            window (int): The rolling window size for extracting the data. Default is 60.
            shrinkage (bool): Whether to apply shrinkage to the covariance matrix. Default is False.

        Returns:
            None
        """
        weight_set = {}

        for time in self.timeseries:
            # Select the pool of stocks based on predictions at the given timestamp
            pool, prediction = self._pool_selection(time, pool_size)
            prediction = prediction.set_index('Symbol')['Prediction'].to_dict()
            order = list(prediction.keys())
            pool_dict = {key: self.price_dict[key] for key in pool}
            end_time = pd.to_datetime(time)
            concated_df = pd.DataFrame()
            for stock, df in pool_dict.items():
                df = self._markowitz_df(df, stock, end_time, window)
                concated_df = pd.concat([concated_df, df], axis=1)
            concated_df = concated_df[order]

            # Handle NaN values
            concated_df = concated_df.dropna()

            if concated_df.empty:
                print(f"No data available for time {time} after dropping NaN values.")
                continue

            cov_mat = concated_df.cov().values
            # Apply shrinkage if True
            if shrinkage:
                cov_mat = np.diag(np.diag(cov_mat))
            order_preds = np.array([prediction[key] for key in order])

    def long_short(self, ret_col='DAYRET', port_name='long_short', top_n=10,
                   long_only=False, short_only=False):
        """
        Implement a long-short strategy for portfolio construction.

        This method constructs a portfolio based on a long-short strategy. It selects the top and bottom stocks
        based on their prediction scores and assigns weights accordingly. The strategy can be configured to be
        long-only, short-only, or both.

        Args:
            ret_col (str): The name of the column containing the returns. Default is 'DAYRET'.
            port_name (str): The name of the portfolio. Default is 'long_short'.
            top_n (int): The number of top stocks to select for long positions and bottom stocks for short positions. Default is 10.
            long_only (bool): If True, the strategy will be long-only. Default is False.
            short_only (bool): If True, the strategy will be short-only. Default is False.

        Returns:
            None
        """
        weight_set = {}

        for time in self.timeseries:
            # Sort predictions by 'Prediction' column in descending order
            prediction = self.predictions[time].sort_values(by='Prediction', ascending=False)
            # Select top K for long and bottom K for short
            long_stocks = prediction.head(top_n)['Symbol'].tolist()
            short_stocks = prediction.tail(top_n)['Symbol'].tolist()
            # Create weights
            weight = {}
            long_weight = 1 / top_n
            short_weight = -1 / top_n
            if long_only:
                for stock in long_stocks:
                    weight[stock] = long_weight
            elif short_only:
                for stock in short_stocks:
                    weight[stock] = short_weight
            else:
                for stock in long_stocks:
                    weight[stock] = long_weight
                for stock in short_stocks:
                    weight[stock] = short_weight
            weight_set[time] = weight

        # Update the weight sets with the new portfolio
        self.weight_sets.update({port_name: weight_set})

    def backtest(self):
        """
        Backtest the constructed portfolios.

        This method calculates the returns of the constructed portfolios over the timeseries.
        It uses the portfolio weights and the true returns to compute the portfolio returns for each timestamp.

        Raises:
            ValueError: If no portfolio weights are found or if a specified portfolio is not found in the weight sets.

        Returns:
            None
        """
        if self.weight_sets is None:
            raise ValueError("No portfolio weights found. Please run a strategy first.")

        else:
            port_names = list(self.weight_sets.keys())
            for port_name in port_names:
                
                if port_name not in self.weight_sets:
                    raise ValueError(f"Portfolio '{port_name}' not found in weight sets.")
                
                weights = self.weight_sets[port_name]
                returns = {}
                # Loop through each timestamp
                for time in self.timeseries:
                    if time not in weights:
                        returns[time] = 0
                        continue

                    weight = weights[time]
                    portfolio_return = 0
                    for key, value in weight.items():
                        df = self.predictions[time]
                        ret = df[df['Symbol'] == key]['True'].values[0]
                        stock_ret = (ret * value) / 5  # Adjust the return calculation as needed
                        portfolio_return += stock_ret

                    returns[time] = portfolio_return

                # Add the returns to the DataFrame
                self.return_sets[port_name] = returns

    def calculate_cumulative_return(self, start_time=None, end_time=None):
        """
        Calculate cumulative returns, Sharpe ratios, and max drawdowns for the portfolios.

        This method calculates the cumulative returns, Sharpe ratios, and max drawdowns for the constructed portfolios
        over a specified time period. It updates the class attributes with the calculated values.

        Args:
            start_time (str or pd.Timestamp, optional): The start time for the calculation. Default is None.
            end_time (str or pd.Timestamp, optional): The end time for the calculation. Default is None.

        Returns:
            None
        """
        # Initialize an empty DataFrame to store cumulative returns
        cum_return_df = pd.DataFrame()

        # Initialize dictionaries to store Sharpe ratios and max drawdowns
        sharpe_ratios = {}
        max_drawdowns = {}

        # Function to find the nearest timestamp
        def find_nearest_timestamp(target_time, timeseries):
            nearest_time = min(timeseries, key=lambda x: abs(x - target_time))
            return nearest_time

        # Set default start and end times if not provided
        if start_time is None:
            start_time = self.timeseries[0]
        else:
            start_time = find_nearest_timestamp(pd.to_datetime(start_time), self.timeseries)

        if end_time is None:
            end_time = self.timeseries[-1]
        else:
            end_time = find_nearest_timestamp(pd.to_datetime(end_time), self.timeseries)

        # Iterate over each portfolio
        for port_name, returns_dict in self.return_sets.items():
            # Convert the return dictionary into a Pandas Series
            returns_series = pd.Series(returns_dict)

            # Filter the returns series based on the start and end times
            filtered_returns_series = returns_series[
                (returns_series.index >= start_time) & (returns_series.index <= end_time)]

            # Calculate cumulative return: (1 + return).cumprod() - 1
            cum_returns = (1 + filtered_returns_series).cumprod() - 1

            # Add the cumulative returns to the DataFrame
            cum_return_df[port_name] = cum_returns

            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # Assuming a 2% risk-free rate, adjust as needed
            excess_returns = filtered_returns_series - risk_free_rate / 252  # Assuming 252 trading days in a year
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            sharpe_ratios[port_name] = sharpe_ratio

            # Calculate max drawdown
            rolling_max = cum_returns.cummax()
            daily_drawdown = rolling_max - cum_returns
            max_drawdown = daily_drawdown.max()
            max_drawdowns[port_name] = max_drawdown

        # Set the index of the DataFrame to be the timestamps (dates)
        cum_return_df.index = pd.to_datetime(cum_return_df.index)
        self.return_df = cum_return_df
        self.sharpe_ratios = sharpe_ratios
        self.max_drawdowns = max_drawdowns

    def plot_cumulative_return(self, portfolios=None, start_time=None, end_time=None):
        """
        Plot the cumulative returns of the selected portfolios.

        This method plots the cumulative returns of the selected portfolios over a specified time period.
        It also displays the Sharpe ratios and max drawdowns for each portfolio in the legend.

        Args:
            portfolios (list, optional): A list of portfolio names to plot. Default is None, which plots all portfolios.
            start_time (str or pd.Timestamp, optional): The start time for the plot. Default is None.
            end_time (str or pd.Timestamp, optional): The end time for the plot. Default is None.

        Returns:
            None
        """
        self.calculate_cumulative_return(start_time=start_time, end_time=end_time)

        # Function to find the nearest timestamp
        def find_nearest_timestamp(target_time, timeseries):
            nearest_time = min(timeseries, key=lambda x: abs(x - target_time))
            return nearest_time

        # Set default start and end times if not provided
        if start_time is None:
            start_time = self.timeseries[0]
        else:
            start_time = find_nearest_timestamp(pd.to_datetime(start_time), self.timeseries)

        if end_time is None:
            end_time = self.timeseries[-1]
        else:
            end_time = find_nearest_timestamp(pd.to_datetime(end_time), self.timeseries)

        # Filter the DataFrame based on the start and end times
        filtered_df = self.return_df[(self.return_df.index >= start_time) & (self.return_df.index <= end_time)]

        # Plot the cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        if not filtered_df.empty and not filtered_df.columns.empty:
            columns_to_plot = portfolios if portfolios else filtered_df.columns
            for column in columns_to_plot:
                if column in filtered_df.columns:
                    ax.plot(filtered_df.index, filtered_df[column],
                            label=f"{column} (Sharpe: {self.sharpe_ratios[column]:.2f},"
                                  f" Max DD: {self.max_drawdowns[column]:.2%})")
                else:
                    print(f"Warning: Portfolio '{column}' not found in data.")

            ax.set_title('Cumulative Returns of Selected Portfolios')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.legend(loc='best')  # Ensure legend is created
            ax.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No data available to plot.")