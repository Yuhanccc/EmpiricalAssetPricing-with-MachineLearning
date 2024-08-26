import tensorflow as tf
from typing import Optional, Tuple

class DefaultModel:
    def __init__(self):
        """Initialize the DefaultModel with pre-defined models."""
        self.default_nn = self.create_default_nn()
        self.default_LSTM: Optional[tf.keras.Model] = None
        self.default_GRU: Optional[tf.keras.Model] = None
        self.default_RNN: Optional[tf.keras.Model] = None

    def create_default_nn(self) -> tf.keras.Model:
        """Create and compile a default neural network model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=32, activation='elu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=64, activation='elu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=64, activation='elu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=32, activation='elu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=16, activation='elu'),
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.R2Score()])
        return model

    def create_default_LSTM(self, input_shape: Tuple[int, ...], learning_rate: float = 0.00001) -> tf.keras.Model:
        """Create and compile a default LSTM model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64,
                                                               recurrent_activation='tanh',
                                                               return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64,
                                                               recurrent_activation='tanh',
                                                               return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64,
                                                               recurrent_activation='tanh',
                                                               return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64,
                                                               recurrent_activation='tanh',
                                                               return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.R2Score()])
        self.default_LSTM = model

    def create_default_RNN(self, input_shape: Tuple[int, ...], learning_rate: float = 0.00001) -> tf.keras.Model:
        """Create and compile a default RNN model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64,
                                                                    return_sequences=True,
                                                                    recurrent_activation='tanh')),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64,
                                                                    recurrent_activation='tanh',
                                                                    return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64,
                                                                    recurrent_activation='tanh',
                                                                    return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64,
                                                                    recurrent_activation='tanh',
                                                                    return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.R2Score()])
        self.default_RNN = model

    def create_default_GRU(self, input_shape: Tuple[int, ...], learning_rate: float = 0.00001) -> tf.keras.Model:
        """Create and compile a default GRU model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64,
                                                              return_sequences=True,
                                                              recurrent_activation='tanh')),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64,
                                                              recurrent_activation='tanh',
                                                              return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64,
                                                              recurrent_activation='tanh',
                                                              return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64,
                                                              recurrent_activation='tanh',
                                                              return_sequences=False)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(units=1, activation='linear',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=[tf.keras.metrics.R2Score()])
        self.default_GRU = model
