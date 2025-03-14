import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime, timedelta
import time

class ImprovedStockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(ImprovedStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply dropout and pass through final layer
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def prepare_data(data, sequence_length=45, target_column='CLOSE', test_size=0.2):
    # Extract basic features
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL']
    
    # Calculate technical indicators
    data['MA7'] = data[target_column].rolling(window=7).mean()
    data['MA14'] = data[target_column].rolling(window=14).mean()
    data['MA30'] = data[target_column].rolling(window=30).mean()
    data['RETURN'] = data[target_column].pct_change()
    data['VOLATILITY'] = data['RETURN'].rolling(window=14).std()
    
    # Calculate RSI
    delta = data[target_column].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values
    data = data.dropna()
    
    # Select features for model input
    selected_features = features + ['MA7', 'MA14', 'MA30', 'RETURN', 'VOLATILITY', 'RSI']
    feature_data = data[selected_features].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, features.index(target_column)])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    return X_train, X_test, y_train, y_test, scaler, features.index(target_column)

def train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=512, patience=30):
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping variables
    best_val_loss = float('inf')
    counter = 0
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            
            train_loss = criterion(model(X_train_tensor), y_train_tensor)
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the best model state
                best_model_state = model.state_dict().copy()
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def train_stock_model(csv_file):
    # Extract ticker from filename
    ticker = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Check if model already exists
    model_dir = os.path.join('trainingModel', ticker)
    model_path = os.path.join(model_dir, 'stock_model.pth')
    scaler_path = os.path.join(model_dir, 'stock_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Model for {ticker} already exists. Skipping training.")
        return None, None
    
    print(f"\nTraining model for {ticker}...")
    
    # Start timer
    start_time = time.time()
    
    # Load data
    data = pd.read_csv(csv_file)
    data.columns = data.columns.str.strip()
    
    if 'TICKER' not in data.columns:
        data['TICKER'] = ticker
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, target_idx = prepare_data(data)
    
    # Create model
    input_size = X_train.shape[2]
    hidden_size = 50
    num_layers = 2
    model = ImprovedStockLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    # Train model
    trained_model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test, epochs=150, batch_size=64, patience=30
    )
    
    # End timer and calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    
    # Determine device type (GPU or CPU)
    device_type = 'GPU' if torch.cuda.is_available() else 'CPU'
    
    # Save training time to appropriate CSV file
    timing_data = {
        'Ticker': ticker,
        'Device': device_type,
        'Training_Time': round(training_time, 2)  # Save training time in seconds
    }
    
    timing_dir = 'training_time'
    
    if not os.path.exists(timing_dir):
        os.makedirs(timing_dir)
    
    timing_csv_path = os.path.join(timing_dir, f'training_time_{device_type}.csv')
    
    if not os.path.exists(timing_csv_path):
        pd.DataFrame([timing_data]).to_csv(timing_csv_path, index=False)
    else:
        existing_data = pd.read_csv(timing_csv_path)
        updated_data = pd.concat([existing_data, pd.DataFrame([timing_data])], ignore_index=True)
        updated_data.to_csv(timing_csv_path, index=False)
    
    print(f"Training time for {ticker}: {training_time:.2f} seconds")
    
    # Save model for future predictions
    save_model_for_prediction(trained_model, scaler, input_size, hidden_size, num_layers, ticker)
    
    # Predict next day's values (hardcoded date modification applied here)
    next_day_values = predict_next_day_full(trained_model, data, scaler, target_idx=target_idx)
    
    print(f"{next_day_values['TICKER']} {next_day_values['PER']} {next_day_values['DATE']} {next_day_values['TIME']} {next_day_values['OPEN']} {next_day_values['HIGH']} {next_day_values['LOW']} {next_day_values['CLOSE']} {next_day_values['VOL']} {next_day_values['OPENINT']}")
    
    prediction_df = pd.DataFrame([next_day_values])
    prediction_dir = os.path.join('predictions')
    
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
        
    prediction_df.to_csv(os.path.join(prediction_dir, f'{ticker}_prediction.csv'), index=False)
    
    print(f"Prediction saved to 'predictions/{ticker}_prediction.csv'")
    
    return trained_model, scaler

def train_all_models():
    # Create trainingModel directory if it doesn't exist
    if not os.path.exists('trainingModel'):
        os.makedirs('trainingModel')
    
    # Get all CSV files in the stock_csv folder
    stock_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_csv")
    csv_files = [os.path.join(stock_csv_dir, f) for f in os.listdir(stock_csv_dir) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Track statistics
    processed_files = 0
    skipped_files = 0
    error_files = 0
    
    # Train model for each CSV file
    for i, csv_file in enumerate(csv_files):
        ticker = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {ticker}")
        
        # Check if model already exists
        model_path = os.path.join('trainingModel', ticker, 'stock_model.pth')
        if os.path.exists(model_path):
            print(f"Model for {ticker} already exists. Skipping training.")
            skipped_files += 1
            continue
            
        try:
            train_stock_model(csv_file)
            processed_files += 1
        except Exception as e:
            print(f"Error training model for {ticker}: {str(e)}")
            error_files += 1
    
    print("\nSummary:")
    print(f"Total files: {len(csv_files)}")
    print(f"Models trained: {processed_files}")
    print(f"Models skipped (already existed): {skipped_files}")
    print(f"Errors encountered: {error_files}")

def evaluate_models():
    # Create results directory if it doesn't exist
    if not os.path.exists('evaluation_results'):
        os.makedirs('evaluation_results')
    
    # Get all model folders
    model_root = 'trainingModel'
    model_folders = [f for f in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, f))]
    
    print(f"Found {len(model_folders)} trained models to evaluate")
    
    # Target date for evaluation (March 7, 2025)
    target_date = '20250307'
    
    # Prepare results dataframe
    results = []
    
    # Loop through each model
    for i, ticker in enumerate(model_folders):
        print(f"\nEvaluating model {i+1}/{len(model_folders)}: {ticker}")
        
        model_dir = os.path.join(model_root, ticker)
        model_path = os.path.join(model_dir, 'stock_model.pth')
        scaler_path = os.path.join(model_dir, 'stock_scaler.pkl')
        
        # Check if model and scaler exist
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print(f"Missing model or scaler for {ticker}, skipping...")
            continue
        
        # Look for the actual data file for the target date
        stock_csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_csv")
        csv_file = os.path.join(stock_csv_dir, f"{ticker}.csv")
        
        if not os.path.exists(csv_file):
            print(f"No data file found for {ticker}, skipping...")
            continue
        
        try:
            # Load the model
            model_info = torch.load(model_path)
            model = ImprovedStockLSTM(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size'],
                num_layers=model_info['num_layers']
            ).to(device)
            model.load_state_dict(model_info['state_dict'])
            model.eval()
            
            # Load the scaler
            scaler = joblib.load(scaler_path)
            
            # Load the data
            data = pd.read_csv(csv_file)
            data.columns = data.columns.str.strip()
            
            # Find the actual data for the target date
            actual_data = data[data['DATE'].astype(str) == target_date]
            
            if len(actual_data) == 0:
                print(f"No actual data found for date {target_date}, skipping...")
                continue
            
            # Get the data up to the day before the target date
            historical_data = data[data['DATE'].astype(str) < target_date].copy()
            
            if len(historical_data) < 45:  # Need at least sequence_length days of data
                print(f"Not enough historical data for {ticker}, skipping...")
                continue
            
            # Calculate technical indicators
            historical_data['MA7'] = historical_data['CLOSE'].rolling(window=7).mean()
            historical_data['MA14'] = historical_data['CLOSE'].rolling(window=14).mean()
            historical_data['MA30'] = historical_data['CLOSE'].rolling(window=30).mean()
            historical_data['RETURN'] = historical_data['CLOSE'].pct_change()
            historical_data['VOLATILITY'] = historical_data['RETURN'].rolling(window=14).std()
            
            # RSI
            delta = historical_data['CLOSE'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            historical_data['RSI'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values
            historical_data = historical_data.dropna()
            
            # Make prediction
            prediction = predict_next_day_full(model, historical_data, scaler)
            
            # Extract actual values
            actual_row = actual_data.iloc[0]
            
            # Print comparison
            print(f"\nPrediction for {ticker} on {target_date}:")
            print(f"Predicted: OPEN={prediction['OPEN']:.4f} HIGH={prediction['HIGH']:.4f} LOW={prediction['LOW']:.4f} CLOSE={prediction['CLOSE']:.4f} VOL={prediction['VOL']:.2f}")
            print(f"Actual:    OPEN={actual_row['OPEN']:.4f} HIGH={actual_row['HIGH']:.4f} LOW={actual_row['LOW']:.4f} CLOSE={actual_row['CLOSE']:.4f} VOL={actual_row['VOL']:.2f}")
            
            # Calculate differences
            open_diff = abs(prediction['OPEN'] - actual_row['OPEN'])
            high_diff = abs(prediction['HIGH'] - actual_row['HIGH'])
            low_diff = abs(prediction['LOW'] - actual_row['LOW'])
            close_diff = abs(prediction['CLOSE'] - actual_row['CLOSE'])
            vol_diff = abs(prediction['VOL'] - actual_row['VOL'])
            
            # Calculate percentage differences
            open_pct = (open_diff / actual_row['OPEN']) * 100 if actual_row['OPEN'] != 0 else float('inf')
            high_pct = (high_diff / actual_row['HIGH']) * 100 if actual_row['HIGH'] != 0 else float('inf')
            low_pct = (low_diff / actual_row['LOW']) * 100 if actual_row['LOW'] != 0 else float('inf')
            close_pct = (close_diff / actual_row['CLOSE']) * 100 if actual_row['CLOSE'] != 0 else float('inf')
            vol_pct = (vol_diff / actual_row['VOL']) * 100 if actual_row['VOL'] != 0 else float('inf')
            
            # Calculate RMSE for the prediction
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            actual_values = np.array([actual_row['OPEN'], actual_row['HIGH'], actual_row['LOW'], actual_row['CLOSE']])
            predicted_values = np.array([prediction['OPEN'], prediction['HIGH'], prediction['LOW'], prediction['CLOSE']])
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            mae = mean_absolute_error(actual_values, predicted_values)
            
            # Print differences
            print(f"\nDifferences:")
            print(f"OPEN: {open_diff:.4f} ({open_pct:.2f}%)")
            print(f"HIGH: {high_diff:.4f} ({high_pct:.2f}%)")
            print(f"LOW: {low_diff:.4f} ({low_pct:.2f}%)")
            print(f"CLOSE: {close_diff:.4f} ({close_pct:.2f}%)")
            print(f"VOL: {vol_diff:.2f} ({vol_pct:.2f}%)")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            
            # Store results
            results.append({
                'Stock': ticker,
                'RMSE': rmse,
                'MAE': mae,
                'Open_Diff': open_diff,
                'Open_Pct': open_pct,
                'High_Diff': high_diff,
                'High_Pct': high_pct,
                'Low_Diff': low_diff,
                'Low_Pct': low_pct,
                'Close_Diff': close_diff,
                'Close_Pct': close_pct,
                'Vol_Diff': vol_diff,
                'Vol_Pct': vol_pct,
                'Predicted_Open': prediction['OPEN'],
                'Predicted_High': prediction['HIGH'],
                'Predicted_Low': prediction['LOW'],
                'Predicted_Close': prediction['CLOSE'],
                'Predicted_Vol': prediction['VOL'],
                'Actual_Open': actual_row['OPEN'],
                'Actual_High': actual_row['HIGH'],
                'Actual_Low': actual_row['LOW'],
                'Actual_Close': actual_row['CLOSE'],
                'Actual_Vol': actual_row['VOL']
            })
            
        except Exception as e:
            print(f"Error evaluating model for {ticker}: {str(e)}")
    
    # Create DataFrame and save results
    if results:
        results_df = pd.DataFrame(results)
        
        # Sort by RMSE (lower is better)
        results_df = results_df.sort_values('RMSE')
        
        # Save to CSV
        results_df.to_csv('evaluation_results/model_evaluation.csv', index=False)
        
        # Create a summary with just stock and error metrics
        summary_df = results_df[['Stock', 'RMSE', 'MAE', 'Close_Pct']]
        summary_df.to_csv('evaluation_results/error_summary.csv', index=False)
        
        print(f"\nEvaluation complete. Results saved to 'evaluation_results/model_evaluation.csv'")
        print(f"Summary saved to 'evaluation_results/error_summary.csv'")
        
        # Print overall statistics
        print(f"\nOverall Statistics:")
        print(f"Average RMSE: {results_df['RMSE'].mean():.4f}")
        print(f"Average MAE: {results_df['MAE'].mean():.4f}")
        print(f"Average Close % Error: {results_df['Close_Pct'].mean():.2f}%")
        print(f"Best performing stock: {results_df.iloc[0]['Stock']} (RMSE: {results_df.iloc[0]['RMSE']:.4f})")
        print(f"Worst performing stock: {results_df.iloc[-1]['Stock']} (RMSE: {results_df.iloc[-1]['RMSE']:.4f})")
    else:
        print("No models were successfully evaluated.")

def predict_next_day_full(model, data, scaler, sequence_length=45, target_idx=3):
    # Get the last sequence_length days of data
    last_data = data.copy()
    
    # Get the features used for prediction
    features = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOL', 'MA7', 'MA14', 'MA30', 'RETURN', 'VOLATILITY', 'RSI']
    last_sequence = last_data[-sequence_length:][features].values
    
    # Scale the data
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Convert to tensor and predict
    input_tensor = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).cpu().numpy()[0, 0]
    
    # Create dummy array for inverse transformation
    dummy = np.zeros((1, scaler.scale_.shape[0]))
    dummy[0, target_idx] = prediction
    
    # Inverse transform to get the actual predicted close price
    predicted_close = scaler.inverse_transform(dummy)[0, target_idx]
    
    # Get the last row for reference
    last_row = data.iloc[-1]
    
    # Calculate price change percentage
    last_close = last_row['CLOSE']
    price_change_pct = (predicted_close - last_close) / last_close
    
    # Estimate other values based on the predicted change
    predicted_open = last_close * (1 + price_change_pct * 0.5)
    predicted_high = max(predicted_close, predicted_open) * (1 + abs(price_change_pct) * 0.2)
    predicted_low = min(predicted_close, predicted_open) * (1 - abs(price_change_pct) * 0.2)
    
    # Ensure high is the highest and low is the lowest
    predicted_high = max(predicted_high, predicted_open, predicted_close)
    predicted_low = min(predicted_low, predicted_open, predicted_close)
    
    # Estimate volume based on price volatility
    predicted_vol = last_row['VOL'] * (1 + abs(price_change_pct) * 2)
    
    # Get next date
    if 'DATE' in last_row:
        # Convert integer date to string, parse it, add one day
        current_date = datetime.strptime(str(int(last_row['DATE'])), '%Y%m%d')
        next_date = current_date + timedelta(days=1)
        # next_date_str = next_date.strftime('%Y%m%d')
        next_date_str = '20250307'
    else:
        # If no DATE column, use current date
        next_date_str = datetime.now().strftime('%Y%m%d')

    
    # Get ticker symbol
    ticker = last_row['TICKER'] if 'TICKER' in last_row else 'A'
    
    # Create prediction row
    prediction_row = {
        'TICKER': ticker,
        'PER': last_row['PER'] if 'PER' in last_row else 'D',
        'DATE': next_date_str,
        'TIME': last_row['TIME'] if 'TIME' in last_row else 0,
        'OPEN': round(predicted_open, 4),
        'HIGH': round(predicted_high, 4),
        'LOW': round(predicted_low, 4),
        'CLOSE': round(predicted_close, 4),
        'VOL': round(predicted_vol, 2),
        'OPENINT': last_row['OPENINT'] if 'OPENINT' in last_row else 0
    }
    
    return prediction_row

def save_model_for_prediction(model, scaler, input_size, hidden_size, num_layers, ticker):
    # Create directory if it doesn't exist
    model_dir = os.path.join('trainingModel', ticker)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model architecture and weights
    model_info = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'state_dict': model.state_dict()
    }
    torch.save(model_info, os.path.join(model_dir, 'stock_model.pth'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'stock_scaler.pkl'))
    
    print(f"Model and scaler for {ticker} saved to '{model_dir}' directory for future predictions.")

def load_model_for_prediction(ticker):
    # Construct path to model directory
    model_dir = os.path.join('trainingModel', ticker)
    
    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"No trained model found for {ticker}")
        return None, None
    
    # Load model
    model_info = torch.load(os.path.join(model_dir, 'stock_model.pth'))
    model = ImprovedStockLSTM(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers']
    ).to(device)
    model.load_state_dict(model_info['state_dict'])
    model.eval()
    
    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, 'stock_scaler.pkl'))
    
    return model, scaler

def main():
    # Create trainingModel directory if it doesn't exist
    if not os.path.exists('trainingModel'):
        os.makedirs('trainingModel')
    
    # Train all models from CSV files in stock_csv folder
    train_all_models()

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train models for all CSV files
    main()
