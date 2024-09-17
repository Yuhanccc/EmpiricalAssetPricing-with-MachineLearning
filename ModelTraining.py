from datetime import timedelta
from typing import Optional, List, Callable, Tuple
import gc
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from tqdm import tqdm


class Trainer:
    def __init__(self, dataset: pd.DataFrame,
                 feature_cols: List[str], target_col: str,
                 date_col: str, symbol_col: str):
        """
        Initialize the ModelTrain class.

        Args:
            dataset (pd.DataFrame): DataFrame containing the data.
            feature_cols (List[str]): List of feature column names.
            target_col (str): Name of the target column.
            date_col (str): Name of the date column used in the index.
            symbol_col (str): Name of the symbol column used in the index.

        Raises:
            ValueError: If required columns or index names are missing from the dataset.
        """
        self.dataset = dataset
        self.validset : Optional[tf.data.Dataset] = None
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.time_series: Optional[np.ndarray] = None
        self.train_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
        self.model: Optional[tf.keras.Model] = None
        self.rnn_attr: Optional[Tuple[bool, int]] = None
        self.train_data : Optional[Tuple] = None
        self.iter_record: Optional[List[Dict[str, Any]]] = None
        self.symbol_lookup: Optional[tf.lookup.StaticVocabularyTable] = None
        self.symbol_reverse_lookup: Optional[tf.lookup.StaticHashTable] = None

        # Validate presence of feature and target columns
        missing_columns = set(self.feature_cols + [self.target_col]) - set(self.dataset.columns)
        if missing_columns:
            raise ValueError(f"The following necessary columns are missing from the dataset: {missing_columns}")

        # Validate presence of date and symbol columns in the index
        missing_indices = {self.date_col, self.symbol_col} - set(self.dataset.index.names)
        if missing_indices:
            raise ValueError(f"The following necessary indices are missing from the dataset index: {missing_indices}")

        @property
        def dataset(self):
            return self._dataset

        @dataset.setter
        def dataset(self, value: pd.DataFrame):
            if not isinstance(value, pd.DataFrame):
                raise ValueError("Dataset must be a pandas DataFrame")
            self._dataset = value

        @property
        def feature_cols(self):
            return self._feature_cols

        @feature_cols.setter
        def feature_cols(self, value: List[str]):
            if not all(isinstance(col, str) for col in value):
                raise ValueError("All feature columns must be strings")
            self._feature_cols = value

    def feature_normalizer(self,
                           normalizer: Optional[Callable[[pd.Series], pd.Series]] = None) -> None:
        """
        Normalizes the feature columns of the dataset by date using the provided normalization function
        and updates the dataset in place.

        Args:
            :param normalizer (Optional[Callable[[pd.Series], pd.Series]]): A function that takes a pandas Series
            and returns a normalized pandas Series. If None, uses the default normalization to [-1, 1].

        Raises:
            ValueError: If the normalization function results in non-numeric data or if any feature column is non-numeric.

        """
        # Group by the index_name (e.g., 'date')
        grouped = self.dataset.groupby(self.date_col)

        # Apply normalization within each group
        for name, group in tqdm(grouped, desc="Normalizing Groups", total=len(grouped)):
            for col in self.feature_cols:
                if pd.api.types.is_numeric_dtype(group[col]):
                    if normalizer is None:
                        # Default normalization to [-1, 1]
                        min_val = group[col].min()
                        max_val = group[col].max()
                        if min_val == max_val:
                            raise ValueError(
                                f"Feature column '{col}' has no variation (min equals max) on {name}. Cannot normalize.")
                        self.dataset.loc[group.index, col] = 2 * ((group[col] - min_val) / (max_val - min_val)) - 1
                    else:
                        # Apply custom normalization function
                        self.dataset.loc[group.index, col] = normalizer(group[col])
                else:
                    raise ValueError(f"Feature column '{col}' must be numeric for normalization.")

    def set_train_window(self, start_date, end_date, start_infer, end_infer):
        """
        Sets the training window for the model by prompting the user to enter the start and end dates.

        This method performs the following steps:
        1. Calls the `_timeseries_indexer` method to ensure the time series index is set.
        2. Prompts the user to input the start and end dates for the training window in the format 'YYYY-MM-DD'.
        3. Validates and processes the input dates using the `_date_checker` method to ensure they are in the correct format.
        4. Checks if the start date is earlier than or equal to the end date. If the start date is later, raises a `ValueError`.
        5. Sets the `train_window` attribute to a tuple containing the start and end dates.
        6. Prints a confirmation message showing the selected training window.

        Returns:
            object: The updated object with the `train_window` attribute set.

        Raises:
            ValueError: If the start date is later than the end date.

        Example:
            self.set_train_window()

            Output:
            Enter the training start date (YYYY-MM-DD): 2024-01-01
            Enter the training end date (YYYY-MM-DD): 2024-06-01
            Training window set from 2024-01-01 to 2024-06-01
        """
        # set time_series instance
        self._timeseries_indexer()

        # Prompt for start and end dates
        train_start_str = start_date
        train_end_str = end_date

        # Process the dates with the user's choices
        train_start = self._date_checker(train_start_str, start_infer)
        train_end = self._date_checker(train_end_str, end_infer)

        # Check the Window
        while train_start > train_end:
            raise ValueError(f"The train_start date {train_start} is greater than the train_end date {train_end}")

        # Set the train window
        self.train_window = (train_start, train_end)
        print(f"Training window set from {train_start} to {train_end}")

    def train_model(self, model: tf.keras.Model,
                    rnn_type: bool = False,
                    seq_length: Optional[int] = None,
                    val_ratio: Optional[float] = None,
                    return_train_record: bool = True, **kwargs):
        """
        Trains the given Keras model using the dataset created from internal data.

        Args:
            model (tf.keras.Model): The Keras model to be trained.
            rnn_type (bool): Flag indicating whether to create RNN-type sequences.
            seq_length (Optional[int]): Length of the sequences for RNN input. Required if rnn_type is True.
            val_ratio (Optional[float]): The ratio of the dataset to be used for validation. If None, no validation dataset is used.
            return_train_record (bool): If True, returns the training history object.
            **kwargs: Additional keyword arguments for model.fit(), e.g., learning rate, callbacks.

        Returns:
            tf.keras.callbacks.History or None: The training history object if return_train_record is True, otherwise None.
        """
        # Create dataset using internal method
        dataset = self._train_tensor_creator(window=self.train_window,
                                             rnn_type=rnn_type, seq_length=seq_length)

        # Apply batching to the dataset
        batch_size = kwargs.pop('batch_size', 512)
        dataset = dataset.batch(batch_size)

        if val_ratio is not None:
            # Split the dataset into training and validation datasets
            train_dataset, val_dataset = self._dataset_split(dataset, val_ratio=val_ratio)
            self.train_data = (train_dataset, val_dataset)

            # Training the model with validation data
            epochs = kwargs.pop('epochs', 15)
            history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, **kwargs)
        else:
            # Training the model without validation data
            epochs = kwargs.pop('epochs', 15)
            history = model.fit(dataset, epochs=epochs, **kwargs)

        # Save the trained model to the instance
        self.model = model
        self.rnn_attr = (rnn_type, seq_length)

        if return_train_record:
            return history
        else:
            return None

    def iter_model(self, rolling_type: str = 'expanding', metric: Callable = r2_score, frozen_layer: Optional[int] = None, **kwargs) -> None:
        """
        Iteratively train the model using transfer learning and record predictions and metrics.

        Args:
            rolling_type (str): Type of rolling window ('expanding' or 'fix').
            metric (Callable): Metric function to evaluate predictions.
            frozen_layer (Optional[int]): Number of layers to keep frozen.
            **kwargs: Additional keyword arguments for model.fit().

        Returns:
            Tuple[List[Tuple[np.datetime64, float]], List[pd.DataFrame], List[dict]]:
                - metric_scores: List of tuples with timestamp and metric score.
                - prediction_records: List of DataFrames with predictions and true values.
                - iter_train_history: List of training history objects.
        """
        self._create_vocab()
        iter_record = []

        iter_begin = np.where(self.time_series == self.train_window[1])[0][0] + 1
        batch_size = kwargs.pop('batch_size', 512)
        epochs = kwargs.pop('epochs', 1)

        # Freeze specified layers
        if frozen_layer is not None:
            num_layers = len(self.model.layers)
            assert 0 < frozen_layer <= num_layers, f"frozen_layer must be between 1 and {num_layers}."

            for layer in self.model.layers[:num_layers - frozen_layer]:
                layer.trainable = False

            for layer in self.model.layers[num_layers - frozen_layer:]:
                layer.trainable = True

        for i, time in enumerate(self.time_series[iter_begin:], start=1):
            time = pd.to_datetime(time)
            dataset = self._iter_tensor_creator(timestamp=time)
            batched_dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            # Get predictions
            pred_l, true_l, symbol_l = self._get_predictions(batched_dataset)
            score = self._calculate_metric_score(np.array(true_l), np.array(pred_l), metric)
            result_df = self._create_result_dataframe(symbol_l, true_l, pred_l, time)
            prediction_summary = self._calculate_prediction_summary(true_l, pred_l)

            # Determine window start
            start = self._get_window_start(rolling_type, time)
            window = (start, time)

            # Create train tensor and fit model
            train_tensor = self._train_tensor_creator(window=window, rnn_type=self.rnn_attr[0], seq_length=self.rnn_attr[1])
            train_tensor = train_tensor.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            # Fine-tune the model on the new data
            try:
                history = self.model.fit(train_tensor, epochs=epochs, verbose=1, **kwargs)
            except Exception as e:
                print(f"Error during model.fit: {e}")
                break

            # clear memory
            del train_tensor
            del dataset
            del batched_dataset
            tf.keras.backend.clear_session()  # Clear TensorFlow session
            gc.collect()  # Manually trigger garbage collection

            # Analyze and record model history
            train_history_summary = self._summarize_train_history(history)
            # Create and store iteration record
            record = {
                'timestamp': time,
                'metric_score': score,
                'prediction_summary': prediction_summary,
                'train_history_summary': train_history_summary,
                'train_window_start': start,
                'train_window_end': time,
                'results': result_df
            }
            iter_record.append(record)
        self.iter_record = iter_record

    def _get_predictions(self, batched_dataset):
        pred_l, true_l, symbol_l = [], [], []
        for batch in batched_dataset:
            features, targets, encoded_symbols = batch
            predictions = self.model.predict(features)
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.reshape(-1)
            decoded_symbols = self.symbol_reverse_lookup.lookup(encoded_symbols)
            pred_l.extend(predictions)
            true_l.extend(targets.numpy())
            symbol_l.extend(decoded_symbols.numpy().astype(str))
        return pred_l, true_l, symbol_l

    def _create_result_dataframe(self,
                                symbols: List[str],
                                true_values: List[float],
                                predictions: List[float],
                                date: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({
            'Symbol': symbols,
            'True': true_values,
            'Prediction': predictions,
            'Date': date
        })

    def _calculate_prediction_summary(self, true_values, predictions):
        errors = np.array(predictions) - np.array(true_values)
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(np.abs(errors)),
            'min_error': np.min(np.abs(errors)),
            'std_error': np.std(errors)
        }

    def _summarize_train_history(self, history):
        return {metric: values[-1] for metric, values in history.history.items()}

    def _timeseries_indexer(self) -> None:
        """
        Extracts the time series data from the dataset by retrieving the unique values of the date column.

        This method retrieves the date values from the specified index level (`self.date_col`), sorts them,
        and stores the unique sorted values in the `self.time_series` attribute.

        Attributes Modified:
            self.time_series (np.ndarray): A sorted NumPy array of unique date values extracted from the dataset's index.
        """
        # Extract the specified index level
        date_index = self.dataset.index.get_level_values(self.date_col)

        # Sort and get unique values
        self.time_series = np.sort(pd.unique(date_index))

    def _date_checker(self, date_str: str, check_type:str) -> pd.Timestamp:
        try:
            date = pd.to_datetime(date_str)
        except ValueError:
            raise ValueError("Invalid date format. Please enter dates in YYYY-MM-DD format.")

        while date not in self.time_series:
            action = check_type.strip().lower()

            if action == "forward":
                date = pd.to_datetime(self._infer_forward(date))
            elif action == "backward":
                date = pd.to_datetime(self._infer_backward(date))
            elif action == "input":
                new_date_str = input("Enter a new date (YYYY-MM-DD): ")
                try:
                    date = pd.to_datetime(new_date_str)
                except ValueError:
                    print("Invalid date format. Please enter dates in YYYY-MM-DD format.")
                    continue  # Re-prompt for the action
            elif action == "quit":
                raise ValueError(f"Date {date_str} not found in the time series.")
            else:
                print("Invalid action. Please choose forward/backward/input/error.")

        return date

    def _infer_backward(self, date: pd.Timestamp) -> pd.Timestamp:
        while True:
            try:
                loc = np.where(self.time_series == date)[0][0]
                return self.time_series[loc]  # Return the timestamp at this location
            except IndexError:
                date -= timedelta(days=1)
                if date < self.time_series.min():  # Prevent infinite loop
                    raise ValueError("Date not found in the time series")

    def _infer_forward(self, date: pd.Timestamp) -> pd.Timestamp:
        while True:
            try:
                loc = np.where(self.time_series == date)[0][0]
                return self.time_series[loc]  # Return the timestamp at this location
            except IndexError:
                date += timedelta(days=1)
                if date > self.time_series.max():  # Prevent infinite loop
                    raise ValueError("Date not found in the time series")

    def _train_tensor_creator(self, window: Tuple[pd.Timestamp, pd.Timestamp],
                              rnn_type: bool = False, seq_length: Optional[int] = None) -> tf.data.Dataset:
        # Extract data for the specified time window
        start, end = window
        self.dataset.sort_index(level=[self.date_col, self.symbol_col], inplace=True)
        train_dataset = self.dataset.loc[start:end, :]

        if rnn_type:
            # Ensure seq_length is provided and valid for RNN-type data
            assert seq_length is not None and seq_length >= 1, \
                "Expected seq_length to be greater than or equal to 1."

            symbols = np.unique(train_dataset.index.get_level_values(self.symbol_col))
            features, targets = [], []

            for symbol in symbols:
                symbol_data = train_dataset.xs(symbol, level=self.symbol_col)
                symbol_feature = symbol_data[self.feature_cols].values
                symbol_target = symbol_data[self.target_col].values

                # Generate sequences for RNN
                for i in range(len(symbol_data) - seq_length + 1):
                    features.append(symbol_feature[i: i + seq_length])
                    targets.append(symbol_target[i + seq_length - 1])

            features = np.array(features)
            targets = np.array(targets)
        else:
            # For non-RNN models, use all data points directly
            features = train_dataset[self.feature_cols].values
            targets = train_dataset[self.target_col].values

        # Create dataset using from_tensor_slices
        tensor_dataset = tf.data.Dataset.from_tensor_slices((features, targets))

        # Optimize data pipeline
        buffer_size = min(len(features), 100000)  # Avoid memory overflow
        tensor_dataset = tensor_dataset.shuffle(buffer_size=buffer_size)

        return tensor_dataset

    def _iter_tensor_creator(self, timestamp: pd.Timestamp) -> tf.data.Dataset:
        """
        Creates a TensorFlow Dataset for a specified timestamp, optionally using sequence data for RNN models.

        Args:
            timestamp (pd.Timestamp): The specific timestamp to extract data for.
            rnn_type (bool): If True, generates sequences of data for RNNs. If False, generates data for the single timestamp.
            seq_length (Optional[int]): The length of sequences to generate for RNN training. Must be greater than or equal to 1 if `rnn_type` is True.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset with features, targets, and symbols, shuffled for training.
        """
        if self.rnn_attr[0]:
            assert self.rnn_attr[1] is not None and self.rnn_attr[1] >= 1, \
                "Expected seq_length to be greater than or equal to 1."

            # Determine the index range for sequence extraction
            iter_end_loc = np.where(self.time_series == timestamp)[0][0]
            iter_start_loc = iter_end_loc - self.rnn_attr[1] + 1

            # source loc to timestamps
            iter_start = self.time_series[iter_start_loc]
            iter_end = self.time_series[iter_end_loc]

            # Extract data within the determined range
            iter_dataset = self.dataset.loc[iter_start:iter_end].copy(deep=True)
            iter_dataset.sort_index(level=[self.symbol_col, self.date_col], inplace=True)

            def generator():
                symbols_unique = np.unique(iter_dataset.index.get_level_values(self.symbol_col))

                for symbol in symbols_unique:
                    symbol_data = iter_dataset.xs(symbol, level=self.symbol_col)

                    if len(symbol_data) >= self.rnn_attr[1]:
                        symbol_feature = symbol_data[self.feature_cols].values
                        symbol_target = symbol_data[self.target_col].values

                        yield (symbol_feature[:self.rnn_attr[1]], symbol_target[self.rnn_attr[1] - 1], symbol)

            output_signature = (
                tf.TensorSpec(shape=(self.rnn_attr[1], len(self.feature_cols)), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )

        else:
            # Extract data for the single timestamp
            iter_dataset = self.dataset.loc[timestamp]

            def generator():
                for index in range(len(iter_dataset)):
                    yield (iter_dataset.iloc[index][self.feature_cols].values,
                           iter_dataset.iloc[index][self.target_col],
                           iter_dataset.index.get_level_values(self.symbol_col)[index])

            output_signature = (
                tf.TensorSpec(shape=(len(self.feature_cols),), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )

        # Create a TensorFlow Dataset including features, targets, and symbols
        tensor_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        # Map symbols to integers using the lookup table
        def map_symbols(features, target, symbol):
            mapped_symbol = self.symbol_lookup.lookup(symbol)
            return features, target, mapped_symbol

        tensor_dataset = tensor_dataset.map(map_symbols)
        tensor_dataset = tensor_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return tensor_dataset

    def _get_window_start(self, rolling_type: str, timestamp: pd.Timestamp) -> np.datetime64:
        """
        Determine the start of the rolling window.

        Args:
            rolling_type (str): Type of rolling window ('expanding' or 'fix').
            timestamp (pd.Timestamp): Timestamp for the end of the rolling window.

        Returns:
            np.datetime64: Start date of the rolling window.
        """
        window_size = np.where(self.time_series == self.train_window[0])[0][0] - \
                      np.where(self.time_series == self.train_window[1])[0][0]
        if rolling_type == 'expanding':
            start = self.train_window[0]
        elif rolling_type == 'fix':
            end_loc = np.where(self.time_series == timestamp)[0][0]
            start = self.time_series[end_loc - window_size]
        else:
            raise ValueError("Rolling type must be either 'expanding' or 'fix'")
        return start

    def _create_vocab(self):
        # Get unique symbols
        symbols_unique = np.unique(self.dataset.index.get_level_values(self.symbol_col))

        # Create symbol lookup table (only once)
        self.symbol_lookup = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(symbols_unique),  # Fixed the variable name
                values=tf.range(tf.size(symbols_unique, out_type=tf.int64), dtype=tf.int64),
            ),
            num_oov_buckets=1  # Handle out-of-vocabulary symbols
        )

        # Create reverse lookup table (only once)
        self.symbol_reverse_lookup = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.range(tf.size(symbols_unique, out_type=tf.int64), dtype=tf.int64),
                values=tf.constant(symbols_unique)
            ),
            default_value=tf.constant("UNKNOWN")
        )

    @staticmethod
    def _dataset_split(dataset: tf.data.Dataset,
                       val_ratio: float = 0.05) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
           Splits a TensorFlow dataset into training and validation sets.

           Args:
               dataset (tf.data.Dataset): The dataset to be split.
               val_ratio (float): The ratio of the dataset to be used for validating.
                                    The remaining data will be used for validation.

           Returns:
               Tuple[tf.data.Dataset, tf.data.Dataset]: A tuple containing the training and validation datasets.
           """
        # Calculate the number of training samples
        dataset_size = dataset.cardinality().numpy()  # Get the size of the dataset
        train_size = int(dataset_size * (1 - val_ratio))

        # Create the training and validation datasets
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)

        return train_dataset, val_dataset

    @staticmethod
    def _model_history_analyzer(history: tf.keras.callbacks.History,
                                timestamp: pd.Timestamp) -> pd.DataFrame:
        # Extract the history dictionary
        history_dict = history.history

        # Create a DataFrame from the history dictionary
        df_history = pd.DataFrame(history_dict)

        # Add an index for epochs
        df_history.index.name = timestamp
        return df_history

    @staticmethod
    def _calculate_metric_score(true_values: np.ndarray, predictions: np.ndarray,
                                metric: Callable) -> float:
        """
        Calculate the metric score.

        Args:
            true_values (np.ndarray): Array of true values.
            predictions (np.ndarray): Array of predicted values.
            metric (Callable): Metric function to calculate score.

        Returns:
            float: Calculated metric score.
        """
        try:
            score = metric(true_values, predictions)
        except Exception as e:
            print(f"Error calculating metric: {e}")
            score = float('nan')
        return score
