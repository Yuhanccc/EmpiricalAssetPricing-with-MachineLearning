# Qlib COPYCAT
## A Share Market's Emprical Asset Pricing with Deep Learning Methods

This repository contains a comprehensive Machine Learning Based Investment Framework designed to facilitate the downloading, processing, and analysis of financial data, as well as the training and evaluation of machine learning models for investment strategies. The framework is organized into five main modules:

1. **DataLoader**: Downloads and merges different price data, including spot prices, adjusted prices, and index prices using the `akshare` library. Returns a dictionary of DataFrames.
2. **Factors**: Contains predefined factor functions and a calculator to apply these factors to stock data. Returns a concatenated DataFrame of factor values.
3. **DefaultModels**: Provides default neural network models, including LSTM, GRU, and RNN, to be used in the ModelTraining module.
4. **ModelTraining**: Trains a base model and uses it to generate predictions iteratively. Returns a dictionary containing predicted returns, true returns, and other information in an `iter_record`.
5. **Portfolios**: Creates and backtests predefined portfolios using predicted returns from ModelTraining. The predefined portfolios include long-short, Markowitz optimized, volatility timing, and return-to-risk timing strategies.

# Modules Overview

### DataLoader

The `DataLoader` class is responsible for downloading and merging different types of price data. It supports loading spot prices, adjusted prices, and index prices. The data is fetched using the `akshare` library and stored in a dictionary of DataFrames.

### Factors

The `Factors` module contains two classes:

- **FactorFunc**: Stores predefined factor functions such as returns, volatility, idiosyncratic volatility, and volume.
- **FactorCalculator**: Applies factor functions to each stock DataFrame in the provided dictionary and returns a concatenated DataFrame of factor values.

### DefaultModels

The `DefaultModels` module provides default neural network models, including:

- A default neural network (NN)
- Long Short-Term Memory (LSTM) network
- Gated Recurrent Unit (GRU) network
- Simple Recurrent Neural Network (RNN)

These models are used in the ModelTraining module for training and prediction.

### ModelTraining

The `ModelTraining` module trains a base model using the provided data and generates predictions iteratively. It returns a dictionary containing predicted returns, true returns, and other relevant information in an `iter_record`.

### Portfolios

The `Portfolios` module creates and backtests predefined portfolios using the predicted returns from the ModelTraining module. The predefined portfolios include:

- **Long-Short**: A strategy that takes long positions in the top-performing stocks and short positions in the bottom-performing stocks.
- **Markowitz Optimized**: A portfolio optimized using Markowitz mean-variance optimization.
- **Volatility Timing**: A strategy that assigns weights inversely proportional to the squared volatility of the stocks.
- **Return-to-Risk Timing**: A strategy that assigns weights proportional to the return-to-risk ratio of the stocks.

## Typical Workflow
![Uploading FlowChart.png…]()


## Sample File
- **Data Download - -  Train Deep Learning Model**: `Samplecode_ModelTraining.ipynb`
- **Portfolio Construction and BackTest**: `Samplecode_BackTest.ipynb`

### Sample Model Training Result (Simple Neural Network Model)
Below is a sample result with model being trained with 40 basic factors (momentum, volatiltiy, liquidity, etc) to make prediction on stock return in future 5 days. The model is trained on data range from 2002 to 2023. Below are some basic results of the model:
- **R2 Score**: Achieved about 6% R2 Score in train set and about 8% in validation set
- **Validation Set**: randomly selected from 2002 to 2023 dataset, 5% of total data, about 500,000 records

![ModelTrain_1](https://github.com/user-attachments/assets/34d6ed88-1e52-473b-b0e9-709c47932bf1)

### Sample Back Test Results
Below is a sample result with model being trained with 40 basic factors (momentum, volatiltiy, liquidity, etc) to make prediction on stock return in future 5 days. The model is then iterated over the Jan to July, 2024. Below are some basic results of the model:
- Detailed result can be found in `Sample_of_Train_Model`
- The model achieved a in-sample r2 score about 8%, and bout 6% in validation set, but poor r2 score in iteration ( between predicted return and actual return )
- The model's accuracy rate (defined as predicture to rise/decline vs. acutally rise/decline) is exceedingly prominent with downside stocks
  - for stocks predicted with 50 lowest return (negative returns), the model can achieve an accuracy rate of 70% in 131 trading days iteration
  - this high accuracy might come from the poor performance of China's A Share Market, it would be much easier to select a stock that crashed
  - on the contrary, for stocks predicted with 50 highest return, the model can only achieve an acucracy of 45% in 131 trading days
  - reasons for the difference can be diverse:
    - insufficient sample of stocks that will rise (poor A share Market)
    - poor performance of A Share Market ( acutally given the performance of A share marekt, the 45% of accuracy is much greater than percentage of stocks that truly goes up in Jan to July, 2024 ,:P)
    - poor trained model, the 70% accuracy in predicting downside stocks are simply luck given by market
    - 
### Simulated Portfolios
To verify the power of the model's prediction result, I run several simulated portfolios and get the following result.
#### General Results
![pic1](https://github.com/user-attachments/assets/ddc0817e-c386-4426-8c88-15a2655d1297)
- For long short portfolio, I long the stocks with top 20 predicted return and short stocks with top 20 predicted loss
 - almost all exceeding performance of long short portfolio comes from the short part
 - pure long 20 portfolio demonstrated negative return
- The power of shorting beats all other portfolios
- All other portfolios demonstrated negative return, which is acceptable, but not realistic in practice
#### Detail About Long Short
![pic2](https://github.com/user-attachments/assets/ae73f38b-b4fd-40f6-a39c-1293c411f2c8)
- Above a look at the performance of long - short portfolio
- The return are purely contributed by short part
  - However this might not be applicable because the resource of sell out in A Share marekt is highly restricted
#### Some Highlights About Volatiltiy Timing
![pic3](https://github.com/user-attachments/assets/7e8b77c2-dc16-4f25-b40f-3e1ee66d4750)
As i noticed the enormous negative impact on portfolio's performance in Jan, 2024, I separated the simulation from Feb 1, 2024, and found prominent performance of Volatility Timing Strategy within 20 stocks with predicted highest return
- The portfolio achieved a return of about 6.2% between Feb 1 and July 30, 2024, Sharpe Ratio of 1.09
- The return of CSI300 is about 4% at the same time range
- A little bit excess earnings, thrilling




## Contributing
Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
