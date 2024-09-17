import pandas as pd
import akshare as ak
import re
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from typing import List, Dict


class DataLoader:
    def __init__(self, stock_codes: List[str], start_date: str, end_date: str):
        """
        Initialize the DataLoader with stock codes, start date, and end date.
        """
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self.index_data: Dict[str, pd.DataFrame] = {}

    @property
    def stock_codes(self) -> List[str]:
        return self._stock_codes

    @stock_codes.setter
    def stock_codes(self, value: List[str]) -> None:
        if not isinstance(value, list):
            raise TypeError('StockCodes must be a list.')
        self._stock_codes = value

    @property
    def start_date(self) -> str:
        return self._start_date

    @start_date.setter
    def start_date(self, value: str) -> None:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError('StartDate must be in the format YYYY-MM-DD.')
        self._start_date = value

    @property
    def end_date(self) -> str:
        return self._end_date

    @end_date.setter
    def end_date(self, value: str) -> None:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            raise ValueError('EndDate must be in the format YYYY-MM-DD.')
        self._end_date = value

    def load_spot_prices(self) -> None:
        """
        Load spot prices for all the stock codes in the specified date range.
        """
        print(f"Loading stock data from {self.start_date} to {self.end_date} for {len(self.stock_codes)} stocks.")
        for stock_code in tqdm(self.stock_codes, desc="Loading stock data"):
            self.stock_data[stock_code] = self._fetch_data(stock_code)

    def load_adjusted_price(self, adj_col_name: str = 'ADJCLOSE') -> None:
        """
        Load adjusted prices for all stock codes. User specifies the adjustment type.
        """
        if not self.stock_data:
            raise ValueError('Load spot prices first using load_spot_prices method.')

        adjust_type = self._prompt_adjust_type()

        for stock_code in tqdm(self.stock_codes, desc="Adjusting stock data"):
            spot_df = self.stock_data.get(stock_code, pd.DataFrame())
            adj_df = self._fetch_data(stock_code, adjust_type)
            if spot_df.empty or adj_df.empty:
                print(f"Data not available for {stock_code}. Skipping...")
                continue

            spot_df.set_index('date', inplace=True)
            adj_df.set_index('date', inplace=True)
            spot_df.index = pd.to_datetime(spot_df.index)
            adj_df.index = pd.to_datetime(adj_df.index)
            adj_df.rename(columns={'close': adj_col_name}, inplace=True)
            adj_df = adj_df[[adj_col_name]]
            self.stock_data[stock_code] = pd.concat([spot_df, adj_df], axis=1, join='inner')
    def _fetch_data(self, stock_code: str, adjust_type: str = None) -> pd.DataFrame:
        """
        Fetch stock price data for a given stock code within the specified date range.
        Optionally, apply adjustment.
        """
        kwargs = {'symbol': stock_code, 'start_date': self.start_date, 'end_date': self.end_date}
        if adjust_type:
            kwargs['adjust'] = adjust_type

        while True:
            try:
                return func_timeout(3, ak.stock_zh_a_daily, kwargs=kwargs)
            except FunctionTimedOut:
                print(f"Timeout fetching data for {stock_code}. Retrying...")
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")
                return pd.DataFrame()

    def _prompt_adjust_type(self) -> str:
        """
        Prompt user for adjustment type and validate input.
        """
        while True:
            adjust_type = input('Enter adjustment type:\n'
                                '1. hfq - Adjust Backward\n'
                                '2. qfq - Adjust Forward\n').strip()
            if adjust_type in ['hfq', 'qfq']:
                return adjust_type
            print('Adjustment type must be either hfq or qfq')

    def load_index(self, index_col_name: str = 'IDXCLOSE') -> None:
        """
        Load index data and merge it with stock data.
        """
        def _stock_grouper(stock_code: str) -> str:
            """
            Determine the market segment for the given stock code.
            """
            market_prefix_dict = {'MainBoard': ['600', '601', '603', '605'],
                                  'SME': ['000', '001', '002', '003'],
                                  'GEM': ['300', '301'],
                                  'STAR': ['688'],
                                  'NEEQ': ['bj']}
            if stock_code[:2] == 'bj':
                return 'NEEQ'
            stock_code = stock_code.strip('shz')
            prefix = stock_code[:3]
            for market, prefixes in market_prefix_dict.items():
                if prefix in prefixes:
                    return market
            return 'Unknown'

        if not self.stock_data:
            raise ValueError('Load spot prices first using load_spot_prices method.')

        self._load_index_data(self.start_date.replace('-', ''),
                         self.end_date.replace('-', ''),
                         col_name = index_col_name)

        for stock_code in tqdm(self.stock_codes, desc="Merging index data"):
            market = _stock_grouper(stock_code)
            if market in self.index_data.keys():
                index_df = self.index_data[market]
                stock_df = self.stock_data.get(stock_code, pd.DataFrame())
                if stock_df.empty:
                    print(f"Data not available for {stock_code}. Skipping...")
                    continue
                merged_df = pd.concat([stock_df, index_df], axis=1, join='inner')
                self.stock_data[stock_code] = merged_df
            else:
                print(f"Market segment for {stock_code} is unknown or not available.")

    def _load_index_data(self, start_date: str, end_date: str, col_name: str) -> None:
        """
        Fetch index data for provided index codes.
        """
        index_codes = {
            'MainBoard': 'sh000001',
            'SME': 'sz399001',
            'GEM': 'sz399006',
            'STAR': 'sh000688',
            'NEEQ': 'sz899050'
        }
        for market, code in index_codes.items():
            index_df = ak.stock_zh_index_daily_em(symbol=code, start_date=start_date, end_date=end_date)
            index_df.rename(columns={'close': col_name}, inplace=True)
            index_df.set_index('date', inplace=True)
            index_df.index = pd.to_datetime(index_df.index)
            index_df = index_df[[col_name]]
            self.index_data[market] = index_df

    @staticmethod
    def load_stock_codes() -> list:
        """
        Load stock codes based on user-selected market(s).

        Returns:
        - list: List of stock codes that match the selected markets.
        """
        # Market definitions
        markets = {
            'MainBoard': ['600', '601', '603', '605'],
            'SME': ['000', '001', '002', '003'],
            'GEM': ['300', '301'],
            'STAR': ['688'],
            'NEEQ': ['bj']
        }

        def prompt_market() -> list:
            """
            Prompt user to select market(s) and validate input.

            Returns:
            - list: List of selected market abbreviations.
            """
            while True:
                input_markets = input(
                    'This function enables download stock codes by market\n'
                    'Please enter the stock markets whose prices you hope to download.\n'
                    'Available markets including:\n'
                    '1. MainBoard - Shanghai Stock Exchange MainBoard\n'
                    '2. SME - Small & Medium-sized Enterprises\n'
                    '3. GEM - Growth Enterprise Market\n'
                    '4. STAR - Sci-Tech innovation board\n'
                    '5. NEEQ - National Equities Exchange and Quotations\n'
                    'Input the abbreviation of stock markets, separated by SPACE: '
                ).split()

                if set(input_markets).issubset(set(markets.keys())):
                    return input_markets
                else:
                    print('Invalid markets selected.')
                    print('Valid market codes are: MainBoard, SME, GEM, STAR, NEEQ')

        def filter_stock_codes(selected_markets: list, market_dict: dict, all_codes: list) -> list:
            """
            Filter stock codes based on selected markets.

            Parameters:
            - selected_markets (list): List of market abbreviations.
            - market_dict (dict): Dictionary of market prefixes.
            - all_codes (list): List of all stock codes.

            Returns:
            - list: Filtered list of stock codes.
            """
            filtered_codes = []
            for market in selected_markets:
                prefixes = market_dict.get(market, [])
                filtered_codes.extend(
                    code for code in all_codes if code.strip('shzbj')[:3] in prefixes
                )
            return filtered_codes

        def handle_neeq(selected_markets: list, market_dict: dict, all_codes: list) -> list:
            """
            Special handling for NEEQ stocks.

            Parameters:
            - selected_markets (list): List of market abbreviations.
            - market_dict (dict): Dictionary of market prefixes.
            - all_codes (list): List of all stock codes.

            Returns:
            - list: Final list of filtered stock codes.
            """
            if 'NEEQ' in selected_markets:
                neeq_codes = [code for code in all_codes if code[:2] == 'bj']
                selected_markets.remove('NEEQ')
                return list(set(neeq_codes + filter_stock_codes(selected_markets, market_dict, all_codes)))
            return filter_stock_codes(selected_markets, market_dict, all_codes)

        # Execute process
        selected_markets = prompt_market()
        all_stock_codes_df = ak.stock_zh_a_spot()
        all_stock_codes = all_stock_codes_df['代码'].tolist()
        filtered_stock_codes = handle_neeq(selected_markets, markets, all_stock_codes)

        return filtered_stock_codes