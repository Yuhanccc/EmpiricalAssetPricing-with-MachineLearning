from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Dict, Callable, Tuple, List

class FactorFunc:

    @staticmethod
    def rets(stock_data: pd.DataFrame, base_col: str, target_col: str) -> pd.DataFrame:
        """
        Calculate the return and add it as a new column in the DataFrame.
        References:
            - Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.
            The Journal of Finance, 19(3), 425-442. DOI: 10.2307/2977928
        """
        stock_data[target_col] = np.log(stock_data[base_col]).diff()
        return stock_data

    @staticmethod
    def volatility(stock_data: pd.DataFrame, base_col: str, target_col: str,
                   window: int) -> pd.DataFrame:
        """
        Calculate the realized volatility and add it as a new column in the DataFrame.
        """
        stock_data[target_col] = stock_data[base_col].rolling(window=window).std()
        return stock_data

    @staticmethod
    def mom(stock_data: pd.DataFrame, ret_col: str, target_col: str, window: int) -> pd.DataFrame:
        """
        Calculate the momentum and add it as a new column in the DataFrame.
        References:
            - Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.
            The Journal of Finance, 48(1), 65-91. DOI: 10.2307/2328882
        """
        stock_data[target_col] = stock_data[ret_col].rolling(window=window).mean()
        return stock_data

    @staticmethod
    def ewm_volatility(stock_data: pd.DataFrame, base_col: str, target_col: str,
                       ewm_param: Optional[float] = None, param_type: str = 'com') -> pd.DataFrame:
        """
        Calculate the exponentially weighted average (EWA) volatility and add it as a new column in the DataFrame.

        Parameters:
            stock_data (pd.DataFrame): The stock data.
            base_col (str): The column used as the base for volatility calculation, expected to be log returns.
            target_col (str): The column where the EWA volatility will be stored.
            ewm_param (Optional[float]): The parameter for the EWM calculation. Default is None.
            param_type (str): The type of EWM parameter to use ('span', 'com', 'halflife', or 'alpha'). Default is 'span'.

        Returns:
            pd.DataFrame: Updated DataFrame with the EWA volatility column added.
        """
        # Select the appropriate EWM parameter type
        if param_type == 'span':
            ewm_volatility = stock_data[base_col].ewm(span=ewm_param).std()
        elif param_type == 'com':
            ewm_volatility = stock_data[base_col].ewm(com=ewm_param).std()
        elif param_type == 'halflife':
            ewm_volatility = stock_data[base_col].ewm(halflife=ewm_param).std()
        elif param_type == 'alpha':
            ewm_volatility = stock_data[base_col].ewm(alpha=ewm_param).std()
        else:
            raise ValueError(
                f"Unsupported param_type: {param_type}. Choose from 'span', 'com', 'halflife', or 'alpha'.")

        stock_data[target_col] = ewm_volatility

        return stock_data

    @staticmethod
    def ivol(df: pd.DataFrame, stk_ret_col: str, idx_ret_col: str, target_col: str, window: int) -> pd.DataFrame:
        """
        Calculate the idiosyncratic volatility (IVOL) of stock returns relative to an index over a rolling window.

        This method performs a rolling regression of stock returns on index returns, computes the standard deviation
        of the residuals (idiosyncratic volatility), and stores the result in a new column.

        Parameters:
            df (pd.DataFrame): DataFrame containing stock and index returns.
            stk_ret_col (str): Column name for stock returns.
            idx_ret_col (str): Column name for index returns.
            target_col (str): Name of the column where the IVOL will be stored.
            window (int): Rolling window size for the regression.

        Returns:
            pd.DataFrame: DataFrame with an additional column containing the idiosyncratic volatility.

        References：
            - Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). The Cross-Section of Volatility and Expected Returns.
            The Journal of Finance, 61(1), 259-299. DOI: 10.1111/j.1540-6261.2006.00836.x
        """
        # Validate inputs
        if stk_ret_col not in df.columns or idx_ret_col not in df.columns:
            raise ValueError(f"Columns '{stk_ret_col}' and '{idx_ret_col}' must exist in the DataFrame.")

        if window < 1 or window > len(df):
            raise ValueError(
                "Window size must be a positive integer and less than or equal to the number of rows in the DataFrame.")

        # Initialize the target column with NaN values
        df[target_col] = np.nan

        # Perform rolling regression
        regression_loop = len(df) - window + 1
        for i in range(regression_loop):
            start = i
            end = i + window
            reg = sm.add_constant(df[idx_ret_col][start:end])
            res = df[stk_ret_col][start:end]
            model = sm.OLS(res, reg, missing='drop')
            model_fit = model.fit()
            ivol = model_fit.resid.std()
            df.at[df.index[end - 1], target_col] = ivol

        return df

    @staticmethod
    def illiquidity(stock_data: pd.DataFrame, ret_col: str,
                    volume_col: str, target_col: str,
                    scale_param: int = 100000000) -> pd.DataFrame:
        """
        Calculate the Amihud Illiquidity measure, which is a proxy for the impact of trading on price.

        Parameters:
            stock_data (pd.DataFrame): The DataFrame containing stock data including returns and volume.
            ret_col (str): The name of the column containing the stock returns.
            volume_col (str): The name of the column containing the trading volume.
            target_col (str): The name of the column where the illiquidity measure will be stored.
            division_param (int, optional): A parameter to scale the volume data (default is 100,000,000).

        Returns:
            pd.DataFrame: The original DataFrame with an additional column for the illiquidity measure.

        References:
            - Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects.
              Journal of Financial Markets, 5(1), 31-56. DOI: 10.1016/S1386-4181(01)00024-6
        """
        # Ensure no division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            stock_data[target_col] = np.where(
                stock_data[volume_col] == 0,
                np.nan,  # Assign NaN where volume is zero
                stock_data[ret_col] / (stock_data[volume_col] / scale_param)
            )

        return stock_data

    @staticmethod
    def mvel(stock_data: pd.DataFrame, price_col: str, shares_col: str,
             target_col: str, scale_param: int = 100000000) -> pd.DataFrame:
        """
        Calculate the Market Value of Equity (MVEL) for each stock in the DataFrame.

        The Market Value of Equity is calculated as the product of stock price and shares outstanding,
        scaled by a parameter for better interpretability.

        Parameters:
            stock_data (pd.DataFrame): The DataFrame containing stock data including price and shares.
            price_col (str): The name of the column containing the stock price.
            shares_col (str): The name of the column containing the number of shares outstanding.
            target_col (str): The name of the column where the Market Value of Equity will be stored.
            scale_param (int, optional): A parameter to scale the Market Value of Equity (default is 100,000,000).

        Returns:
            pd.DataFrame: The original DataFrame with an additional column for the Market Value of Equity.

        References:
            - Banz, R. W. (1981). The Relationship Between Return and Market Value of Common Stocks.
              Journal of Financial Economics, 9(1), 3-18. DOI: 10.1016/0304-405X(81)90018-0
        """
        # Ensure there are no missing values in price or shares columns
        if price_col not in stock_data.columns or shares_col not in stock_data.columns:
            raise ValueError(f"Columns '{price_col}' or '{shares_col}' not found in DataFrame.")

        # Compute the Market Value of Equity (MVEL)
        stock_data[target_col] = (stock_data[shares_col] * stock_data[price_col]) / scale_param

        return stock_data

    @staticmethod
    def maxret(stock_data: pd.DataFrame, ret_col: str, target_col: str, window: int = 5) -> pd.DataFrame:
        """
        Calculate the rolling maximum returns over a specified window period.

        This function computes the maximum return over a rolling window, which can be useful for identifying
        stocks that exhibit lottery-like behavior, as discussed in the literature.

        Parameters:
            stock_data (pd.DataFrame): The DataFrame containing stock returns.
            ret_col (str): The name of the column containing the returns.
            target_col (str): The name of the column where the rolling maximum returns will be stored.
            window (int, optional): The rolling window size to compute the maximum return (default is 5).

        Returns:
            pd.DataFrame: The original DataFrame with an additional column for the rolling maximum returns.

        References:
            - Bali, T. G., Cakici, N., & Whitelaw, R. F. (2011). Maxing Out: Stocks as Lotteries and the Cross-Section of Expected Returns.
              Journal of Financial Economics, 99(2), 427-446. DOI: 10.1016/j.jfineco.2010.08.001
        """
        # Calculate rolling maximum returns
        stock_data[target_col] = stock_data[ret_col].rolling(window=window).max()

        return stock_data

    @staticmethod
    def turnover(stock_data: pd.DataFrame, turn_col: str, target_col: str, window: int = 10) -> pd.DataFrame:
        """
        Calculate the rolling mean of turnover over a specified window period.

        Turnover is a measure of the trading activity of a stock, and the rolling mean helps in smoothing out
        short-term fluctuations to analyze trends.

        Parameters:
            stock_data (pd.DataFrame): The DataFrame containing stock turnover data.
            turn_col (str): The name of the column containing the turnover values.
            target_col (str): The name of the column where the rolling mean turnover will be stored.
            window (int, optional): The rolling window size to compute the mean turnover (default is 10).

        Returns:
            pd.DataFrame: The original DataFrame with an additional column for the rolling mean turnover.

        References:
            - Datar, V. T., Naik, N. Y., & Radcliffe, R. (1998). Liquidity and Stock Returns: An Alternative Test.
              Journal of Financial Markets, 1(2), 203-219. DOI: 10.1016/S1386-4181(97)00007-8
        """
        # Calculate rolling mean turnover
        stock_data[target_col] = stock_data[turn_col].rolling(window=window).mean()

        return stock_data

    @staticmethod
    def factor_diff(stock_data: pd.DataFrame, base_col: str, target_col: str,
                      window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculates the change in values of the base column over a specified window period and stores it in the target column.

        Parameters:
            stock_data (pd.DataFrame): The DataFrame containing the stock data.
            base_col (str): The column name in stock_data on which the difference will be computed.
            target_col (str): The column name where the computed differences will be stored.
            window (Optional[int]): The number of periods over which to calculate the difference. If None, calculates the simple difference.

        Returns:
            pd.DataFrame: Updated DataFrame with the target_col containing the computed differences.

        Raises:
            ValueError: If window is less than 1 or not an integer.
        """
        if window is not None:
            if not isinstance(window, int) or window < 1:
                raise ValueError("Window must be an integer greater than or equal to 1.")
            stock_data[target_col] = stock_data[base_col].diff(periods=window)
        else:
            stock_data[target_col] = stock_data[base_col].diff()

        return stock_data

class FactorCalculator:
    def __init__(self, StockDict: Dict[str, pd.DataFrame]):
        self.StockDict = StockDict
        self.dataset: Optional[pd.DataFrame] = None # Initialize dataset attribute
        self.label_window: Optional[int] = None  # Initialize label_window attribute
        self.label_col: Optional[str] = None # Initialize label_col attribute

    @property
    def StockDict(self) -> Dict[str, pd.DataFrame]:
        return self._StockDict

    @StockDict.setter
    def StockDict(self, StockDict: Dict[str, pd.DataFrame]) -> None:
        if not isinstance(StockDict, dict):
            raise TypeError('StockDict must be a dictionary of DataFrames.')
        if len(StockDict) == 0:
            raise ValueError('StockDict cannot be empty.')
        for key, value in StockDict.items():
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f'All values in StockDict must be pandas DataFrames. Error with key: {key}')
        self._StockDict = StockDict

    def get_columns(self) -> list:
        """
        Returns the columns of the DataFrames in StockDict, assuming all DataFrames have the same columns.

        Returns:
            list: A list of column names from one of the DataFrames in StockDict.
        """
        # Retrieve the first DataFrame in the StockDict and return its columns
        first_key = next(iter(self.StockDict))
        return list(self.StockDict[first_key].columns)

    def get_indexcols(self) -> list:
        """
        Retrieve the index column names from the first DataFrame in StockDict.

        This method accesses the first DataFrame stored in the StockDict dictionary and
        returns a list of the names of the index columns. This can be useful for understanding
        the structure of the DataFrames being processed or ensuring consistency across multiple DataFrames.

        Returns:
            list: A list of index column names from the first DataFrame in StockDict.
        """
        # Retrieve the first DataFrame in the StockDict and return its columns
        first_index = next(iter(self.StockDict))
        return list(self.StockDict[first_index].index.names)

    def delete_columns(self, columns_to_delete: List[str]) -> None:
        """
        Deletes specified columns from all DataFrames in StockDict.

        Parameters:
            columns_to_delete (list[str]): A list of column names to delete from each DataFrame.
        """
        for stock_code, stock_data in self.StockDict.items():
            # Drop the specified columns, if they exist in the DataFrame
            self.StockDict[stock_code] = stock_data.drop(columns=columns_to_delete, errors='ignore')

    def create_factor(self, index_name: str = 'date',
                      factor_func: Callable[..., pd.DataFrame] = None,
                       **kwargs) -> None:
        """
        Create base metrics for each DataFrame in StockDict using a specified metric function.

        Parameters:
            index_name (str): The name of the index column to sort by (default 'date').
            metric_function (Callable): A function from Metrics class to calculate the metric.
            **kwargs: Additional keyword arguments to be passed to the metric_function.
        """
        self._iterate_stocks(
            lambda stock_data: factor_func(stock_data, **kwargs),
            index_name = index_name
        )

    def create_label(self, ret_col: str, target_col: str, window: int, index_name: str = 'date') -> None:
        """
        Create a target column in each DataFrame within StockDict based on the rolling window sum of the return column.

        This function iterates over each DataFrame in StockDict, calculates the rolling sum of the `ret_col`
        over the specified `window`, and stores the result in `target_col`. The rolling sum is then shifted
        by the window size.

        Parameters:
            ret_col (str): The column name in each DataFrame containing return values.
            target_col (str): The column name where the target values will be created.
            window (int): The size of the rolling window.
            index_name (str, optional): The name of the index column to sort by (default is 'date').

        Raises:
            ValueError: If `ret_col` is not found in any DataFrame, or if the window size is invalid.
        """
        self.label_col = target_col
        self.label_window = window

        def label_function(stock_data: pd.DataFrame) -> pd.DataFrame:
            # Validate columns
            if ret_col not in stock_data.columns:
                raise ValueError(f"Column '{ret_col}' not found in the DataFrame.")

            # Validate window size
            if window < 1:
                raise ValueError("Window size must be greater than or equal to 1.")
            if window >= len(stock_data):
                raise ValueError("Window size must be less than the number of rows in the DataFrame.")

            # Create the target column with rolling window sum and shift
            stock_data[target_col] = stock_data[ret_col].rolling(window=window).sum().shift(-window)

            return stock_data

        self._iterate_stocks(label_function, index_name=index_name)

    def _iterate_stocks(self, operation: Callable[[pd.DataFrame], pd.DataFrame], index_name: str) -> None:
        """
        Iterate over each DataFrame in StockDict and apply the given operation.

        Parameters:
            operation (Callable[[pd.DataFrame], pd.DataFrame]): A function to apply to each DataFrame.
            index_name (str): The name of the index column to sort by (default 'date').
        """
        for StockCode, StockData in tqdm(self.StockDict.items(), desc="Processing Stocks"):
            try:
                StockData.sort_index(level=index_name, ascending=True, inplace=True)
                StockData = operation(StockData)
                self.StockDict[StockCode] = StockData
            except Exception as e:
                print(f'Error {e} occur when processing {StockCode}')

    def concat(self, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process each DataFrame in StockDict by adding a 'stockcode' column, setting 'stockcode' as the second-level index,
        and then concatenating them. Also, extract the last few rows of each DataFrame based on label_window
        and concatenate them.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - The concatenated DataFrame of all processed stock data.
                - The concatenated DataFrame of the last `label_window` rows from each stock data.
        """
        training_data = []
        prediction_data = []

        for stock_code, stock_data in self.StockDict.items():
            # Add 'stockcode' column with the stock code as the value
            stock_data['stockcode'] = stock_code

            # Set 'stockcode' as the second-level index
            df = stock_data.copy(deep=True)
            df.set_index('stockcode', append=True, inplace=True)

            # Append the processed DataFrame to the training_data list
            training_data.append(df)

            # Extract the last few rows based on label_window for predictions
            last_rows = df.sort_index(level=date_col).iloc[-self.label_window:]
            last_rows.drop(columns=self.label_col, inplace=True)

            # Drop rows where the label column is NaN in prediction data
            prediction_data.append(last_rows)

        # Concatenate all training data along the 'date' and 'stockcode' indices, dropping rows with NaNs
        training_data_concat = pd.concat(training_data, axis=0).dropna(how='any', axis=0)

        # Concatenate all prediction data, dropping any rows with NaNs
        prediction_data_concat = pd.concat(prediction_data, axis=0).dropna(how='any', axis=0)

        return training_data_concat, prediction_data_concat

