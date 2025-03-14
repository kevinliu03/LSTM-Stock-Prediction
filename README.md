# Stock Price Prediction with LSTM Neural Networks

## Overview
This Python script implements a Long Short-Term Memory (LSTM) neural network model for predicting stock prices. The model uses historical stock data to forecast future prices, incorporating technical indicators to improve prediction accuracy.

## Library Import
To install the required libraries, run:
```pip install torch numpy pandas scikit-learn joblib```

## Function Explanations

### ImprovedStockLSTM
- A custom LSTM neural network class for stock price prediction.
- Implements a stacked LSTM architecture with configurable hidden size and layers.
- Uses dropout (20%) for regularization to prevent overfitting.
- Includes a fully connected output layer to generate predictions.
- The `forward` method handles initialization of hidden states and processing of input sequences.

### prepare_data
- Transforms raw stock data into a format suitable for LSTM training.
- Calculates technical indicators (MA7, MA14, MA30, RSI, Volatility).
- Normalizes all features to a 0-1 range using MinMaxScaler.
- Creates sequences of 45 days to predict the next day's price.
- Splits data into training (80%) and testing (20%) sets.
- Returns prepared data tensors, scaler, and target index.

### train_model
- Handles the complete training process for the LSTM model.
- Converts data to PyTorch tensors and moves them to the appropriate device (CPU/GPU).
- Implements mini-batch training with configurable batch size.
- Uses Adam optimizer and MSE loss function.
- Applies learning rate scheduling to reduce learning rate when progress plateaus.
- Implements early stopping with patience=30 to prevent overfitting.
- Applies gradient clipping to prevent exploding gradients.
- Tracks and saves the best model based on validation performance.

### train_stock_model
- Orchestrates the complete training process for a single stock.
- Extracts ticker symbol from the filename.
- Checks if a model already exists to avoid redundant training.
- Loads and prepares data using `prepare_data`.
- Creates an LSTM model with appropriate parameters (`input_size`, `hidden_size`, etc.).
- Calls `train_model` to handle the actual training process.
- Saves the trained model and scaler for future use using `save_model_for_prediction`.
- Generates predictions for the next trading day using `predict_next_day_full`.

### train_all_models
- Automates the training process for multiple stocks.
- Processes all CSV files in the `stock_csv` directory.
- Tracks statistics on successful, skipped, and failed training attempts.
- Provides a summary of the overall training process.

### evaluate_models
Evaluates the performance of trained models against actual historical data:
- Tests each model by predicting a specific target date (e.g., March 7, 2025).
- Calculates error metrics such as RMSE, MAE, and percentage differences between predicted vs. actual values.
- Generates detailed reports and identifies best/worst performing models.

### predict_next_day_full
- Generates predictions for the next trading day using the trained model.
- Takes the last sequence of days from historical data.
- Processes input through the model to predict closing price.
- Estimates related values (open, high, low, volume) based on predicted close.
- Calculates the next day's date for the prediction.
- Returns a complete prediction row in the standard OHLCV format.

### save_model_for_prediction
- Saves the trained model and associated data for future use.
- Creates a directory structure organized by ticker symbol.
- Saves model architecture parameters and learned weights.
- Saves the scaler used for normalization (essential for proper prediction).
- Enables future predictions without retraining.

### load_model_for_prediction
- Loads a previously trained model for making predictions.
- Reconstructs the model architecture from saved parameters.
- Loads the trained weights into the model.
- Retrieves the associated scaler for proper data transformation.
- Returns the model ready for making predictions.

### main
The entry point function that orchestrates everything:
- Creates necessary directories (e.g., `trainingModel`).
- Calls `train_all_models` to process all stock data files in bulk from `stock_csv`.
- Sets up GPU/CPU environment for efficient training.

## Usage

To train models for all stock CSV files in your `stock_csv` directory:
