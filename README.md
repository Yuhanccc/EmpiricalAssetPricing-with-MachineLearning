# Empirical Asset Pricing with Machine Learning in China's A Share Market

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
Two files are created for a sample usage
- Sample_of_Train_Model : this sample file illustrate fields including: data downloading, factor calculation, model train, model iteration
- Sample_of_Construct_Portfolios: this sample file illustrate how to construct more sophisticated portfolios with given prediction results of the model. Pre-defined portfolios including:
  - Markowitz Portfolio : Simple Mean Variance Optimization with return predicted by earlier trained model and historical volatility
  - Markowitz Portfolio with Shrinkage: same as above except using shrinkage method on the covaraince matrix, reduce covariance matrix to diagonal matrix
  - Volatility Timing Portfolio: proposed by Chris Kirby and Barbara Ostdiek, portfolio weights are inversely proportional to realized volatility
  - Risk To Reward Timing Portfolio: portfolio weight is inversely proportional to the ( predicted return / realized volatility ratio )
  - Long Short Portfolio: long top k stocks with highest predicted return and short top k stocks with lowest predicted return, along with long only and short only options

## Contributing
Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
