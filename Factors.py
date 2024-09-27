from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional, Dict, Callable, Tuple, List
from multiprocessing import Pool, cpu_count


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
    def volume(stock_data: pd.DataFrame, base_col: str, target_col: str,
               window: int = 5) -> pd.DataFrame:
        if not isinstance(window, int) or window < 2:
            raise ValueError("Window must be an integer greater than or equal to 2.")
        stock_data[target_col] = stock_data[base_col].rolling(window=window).mean() / 100000000

        return stock_data


class FactorCalculator:
    def __init__(self, StockDict: Dict[str, pd.DataFrame]):
        self.StockDict = StockDict
        self.dataset: Optional[pd.DataFrame] = None  # Initialize dataset attribute
        self.label_window: Optional[int] = None  # Initialize label_window attribute
        self.label_col: Optional[str] = None  # Initialize label_col attribute

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

    def create_factor(self,
                      index_name: str = 'date',
                      factor_func: Callable[..., pd.DataFrame] = None,
                      use_multiprocessing: bool = False,
                      **kwargs) -> None:
        """
        Create base metrics for each DataFrame in StockDict using a specified metric function.

        Parameters:
            index_name (str): The name of the index column to sort by (default 'date').
            factor_func (Callable): A function to calculate the factor for each DataFrame.
            use_multiprocessing (bool): Whether to use multiprocessing. Default is False.
            **kwargs: Additional keyword arguments to be passed to the factor_func.
        """
        self._iterate_stocks(
            operation=factor_func,
            index_name=index_name,
            use_multiprocessing=use_multiprocessing,
            **kwargs
        )

    def _iterate_stocks(self,
                        operation: Callable[[pd.DataFrame, Dict], pd.DataFrame],
                        index_name: str,
                        use_multiprocessing: bool = False,
                        **kwargs) -> None:
        """
        Iterate over each DataFrame in StockDict and apply the given operation.

        Parameters:
            operation (Callable[[pd.DataFrame, Dict], pd.DataFrame]): A function to apply to each DataFrame.
            index_name (str): The name of the index column to sort by (default 'date').
            use_multiprocessing (bool): Whether to use multiprocessing. Default is False.
            **kwargs: Additional keyword arguments to be passed to the operation function.
        """
        if use_multiprocessing:
            num_workers = min(cpu_count(), len(self.StockDict))
            stock_dict_items = list(self.StockDict.items())
            chunks = [stock_dict_items[i::num_workers] for i in range(num_workers)]

            # Create a tqdm instance for progress tracking
            with tqdm(total=len(stock_dict_items), desc="Processing Stocks") as pbar:
                with Pool(processes=num_workers) as pool:
                    results = []
                    for result in pool.imap_unordered(FactorCalculator._process_chunk,
                                                      [(chunk, operation, index_name, kwargs) for chunk in chunks]):
                        results.extend(result)
                        pbar.update(len(result))  # Update the progress bar for each chunk processed

            # Combine the results back into StockDict
            self.StockDict = {key: value for key, value in results}
        else:
            for StockCode, StockData in tqdm(self.StockDict.items(), desc="Processing Stocks"):
                try:
                    StockData.sort_index(level=index_name, ascending=True, inplace=True)
                    StockData = operation(StockData, **kwargs)
                    self.StockDict[StockCode] = StockData
                except Exception as e:
                    print(f'Error {e} occurred when processing {StockCode}')

    @staticmethod
    def _process_chunk(
            args: Tuple[List[Tuple[str, pd.DataFrame]], Callable[[pd.DataFrame, Dict], pd.DataFrame], str, Dict]) -> \
    List[Tuple[str, pd.DataFrame]]:
        chunk, operation, index_name, kwargs = args
        processed_chunk = []

        for StockCode, StockData in chunk:
            try:
                StockData.sort_index(level=index_name, ascending=True, inplace=True)
                StockData = operation(StockData, **kwargs)
                processed_chunk.append((StockCode, StockData))
            except Exception as e:
                print(f'Error {e} occurred when processing {StockCode}')

        return processed_chunk

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