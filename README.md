# Stock Price Prediction with LSTM

This repository contains code for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. The goal is to predict the next day's closing price based on the previous five days' data, including the opening price, high price, low price, and closing price.

The code is implemented in Python using the PyTorch library and includes a custom dataset class, a neural network model, and training code. Additionally, it mentions the integration of a tweet sentiment indicator, although the implementation details for this aspect are not provided in the code snippet.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- pandas
- numpy
- matplotlib

You can install these packages using pip by running the following command:

```
pip install torch pandas numpy matplotlib
```

## Dataset

The dataset used for training the model is a CSV file containing historical stock price data for a specific stock (in this case, AAPL). The file should have columns for Date, Open, High, Low, Close, and Volume. The dataset file path should be provided as an argument to the `StockDataset` class.

The `StockDataset` class reads the CSV file, preprocesses the data by removing the Date and Volume columns, and provides the data in the required format for training the LSTM model.

## Model Architecture

The neural network model used for stock price prediction is defined in the `Net` class. The model consists of an LSTM layer with an input size of 5 (representing the five features: Open, High, Low, Close, Volume), followed by multiple LSTM layers, ReLU activation function, and a fully connected layer.

The model takes a sequence of five days' data as input and predicts the closing price for the next day. The LSTM layers capture the temporal dependencies in the input sequence, allowing the model to learn patterns and make accurate predictions.

## Training

The training code uses the `StockDataset` class to load the dataset and create a data loader for batching the data. It then initializes the LSTM model, defines the loss function (Mean Squared Error), and sets up the Adam optimizer for training.

The training is performed over multiple epochs, with each epoch iterating through the dataset in batches. The model parameters are updated using backpropagation and gradient descent optimization. The loss is calculated and printed for every 100th batch during training.

After training, the model parameters are saved to a file named `stock_net.pth` using the `torch.save()` function. This allows the trained model to be loaded and used for making predictions in the future.

## Future Work - Tweet Sentiment Indicator

The code provided mentions the integration of a tweet sentiment indicator, but the implementation details are not included. Adding a sentiment indicator to the stock price prediction model can provide additional information to improve the prediction accuracy.

To implement the tweet sentiment indicator, you would need to gather relevant tweets related to the stock and analyze their sentiment using natural language processing techniques. The sentiment scores or labels can then be incorporated into the input data for the LSTM model. This can be done by combining the sentiment scores with the existing features or by training a separate sentiment model and using its predictions as an additional input to the stock price prediction model.

It's worth noting that the tweet sentiment indicator implementation is not provided in the code snippet, and you would need to design and develop it separately based on your specific requirements and the available data.

## Conclusion

This repository demonstrates how to use LSTM neural networks for stock price prediction based on historical data. The code provides a custom dataset class, a model architecture using LSTM layers, and training code to train the model. The implementation of a tweet sentiment indicator is mentioned as future work but not included in the code snippet.

Feel free to use this code as a starting point and customize it according to your specific needs.
