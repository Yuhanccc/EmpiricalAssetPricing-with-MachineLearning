# Factor Model and Training System for Financial Data

## Overview
This project implements a comprehensive system for loading financial data, calculating factors, and training models for financial analysis and prediction. It's designed to streamline the process of developing and testing factor models in quantitative finance, leveraging advanced machine learning techniques.

## Project Structure
The project consists of the following main components:

1. `DataLoader.py`: Handles the loading and preprocessing of financial data.
2. `DefaultModel.py`: Defines multiple pre-defined machine learning models.
3. `FactorCalculator.py`: Calculates various financial factors from the input data.
4. `ModelTraining.py`: Manages the process of training the factor models.

## Features
- Efficient data loading and preprocessing for financial datasets
- Flexible factor calculation system
- Multiple pre-defined machine learning models:
  - Neural Networks (NN)
  - Recurrent Neural Networks (RNN)
  - Long Short-Term Memory networks (LSTM)
  - Gated Recurrent Units (GRU)
  - And more...
- All models implemented using the TensorFlow framework
- Comprehensive model training pipeline
- Easy model selection and hyperparameter tuning

## Installation
To use this system, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yuhanccc/TimeSeriesForecast.git
cd factor-model-training-system
pip install -r requirements.txt
``` 


## Usage
Here's a basic example of how to use the system:
```python
from DataLoader import DataLoader
from FactorCalculator import FactorCalculator
from DefaultModel import DefaultModel
from ModelTraining import ModelTrainer
```
### Load and preprocess data
```python
# Initialize DataLoader
Loader = DataLoader(StockCodes, start_date='2002-01-01', end_date='2024-07-31')
# Load Spot Price
Loader.load_spot_prices()
# Load Adjusted Price
Loader.load_adjusted_price()
# Load Index Price
Loader.load_index()
```
### Calculate factors
```python
# Initialize Factor Handler
Calculator = FactorCalculator(StockDict)
# Create Some Features
Calculator.create_factor(factor_func=FactorFunc.rets,base_col = 'ADJCLOSE', target_col = 'DAYRET')
Calculator.create_factor(factor_func=FactorFunc.rets,base_col = 'IDXCLOSE', target_col = 'IDXRET')
# Realised Volatility Factors
Calculator.create_factor(factor_func=FactorFunc.volatility, base_col = 'DAYRET',
                         target_col = 'RV5', window = 5)
Calculator.create_factor(factor_func=FactorFunc.volatility, base_col = 'DAYRET',
                         target_col = 'RV10', window = 10)
# Create Label
Calculator.create_label(ret_col = 'DAYRET', target_col = 'Target', window = 3)
# Concat dataframe for training
Train, Pred = Calculator.concat()
```
### Initialize the model (e.g., LSTM)
```python
# Initialse Models class
Models = DefaultModel()

# Use default LSTM, default 20 steps
input_shape = (20, len(feature_cols))
Models.create_default_LSTM(input_shape=input_shape)
# Initialise Trainer
Training =Trainer(dataset=Train,
                  feature_cols = feature_cols, target_col = target_col,
                  date_col = date_col, symbol_col = symbol_col)
# Set Training Window
Training.set_train_window()
```
### Train Model
```python
# Define an early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # The metric to monitor
    patience=3,          # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Whether to restore the model weights from the epoch with the best value of the monitored metric
    verbose=1            # Verbosity mode, 1 = show messages when early stopping is triggered
)
# Train the model
Train_rec = Training.train_model(model = Models.default_LSTM,
                                 rnn_type=True, seq_length=20,
                                 val_ratio=0.05, return_train_record=True, epochs=50,
                                 callbacks=[early_stopping])
```
### Use the trained model for predictions
```python
predictions = Training.model.predict(data)
```
### Iterate the model and make predictions
```python
train_model.iter_model()
predictions = train_model.return_predictions
```

## Model Types
Our `DefaultModel` class supports various types of neural networks, all implemented using TensorFlow:

- Feedforward Neural Networks (NN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory networks (LSTM)
- Gated Recurrent Units (GRU)
- Convolutional Neural Networks (CNN)
- ...(other model types to be added)

You can easily select and configure these models by specifying the `model_type` and relevant parameters when initializing the `DefaultModel`.

## Contributing
Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
