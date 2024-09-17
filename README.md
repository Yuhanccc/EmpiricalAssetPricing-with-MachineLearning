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
- `Sample_of_Train_Model` : this sample file illustrate fields including: data downloading, factor calculation, model train, model iteration
- `Sample_of_Construct_Portfolios`: this sample file illustrate how to construct more sophisticated portfolios with given prediction results of the model. Pre-defined portfolios including:
  - Markowitz Portfolio : Simple Mean Variance Optimization with return predicted by earlier trained model and historical volatility
  - Markowitz Portfolio with Shrinkage: same as above except using shrinkage method on the covaraince matrix, reduce covariance matrix to diagonal matrix
  - Volatility Timing Portfolio: proposed by Chris Kirby and Barbara Ostdiek, portfolio weights are inversely proportional to realized volatility
  - Risk To Reward Timing Portfolio: portfolio weight is inversely proportional to the ( predicted return / realized volatility ratio )
  - Long Short Portfolio: long top k stocks with highest predicted return and short top k stocks with lowest predicted return, along with long only and short only options

## Sample Results
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
- The portfolio achieved a return of about 6.2% between Feb 1 and July 30, 2024
- The return of CSI300 is about 4% at the same time range
- A little bit excess earnings, thrilling




## Contributing
Contributions to improve the system are welcome. Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
