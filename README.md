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

'''bash
git clone https://github.com/yuhanccc/TimeSeriesForecast.git
cd factor-model-training-system
pip install -r requirements.txt
'''

