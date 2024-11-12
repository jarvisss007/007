import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def perform_ml_analysis(hist):
    """Perform machine learning analysis on stock data."""
    if hist.empty:
        raise ValueError("Historical data is empty. Cannot perform machine learning analysis.")

    # Prepare features and target
    features = hist[['Close', 'MA10', 'MA50', 'RSI', 'MACD']].dropna().values
    target = hist['Close'].dropna().values

    # Feature Scaling
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target.reshape(-1, 1))

    # Split data into training and testing sets
    train_size = int(len(scaled_features) * 0.8)
    train_features, train_target = scaled_features[:train_size], scaled_target[:train_size]
    test_features, test_target = scaled_features[train_size:], scaled_target[train_size:]

    # Train MLP Model
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size=0.2, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=500, random_state=42)
    model.fit(X_train, y_train.flatten())

    # Predict and evaluate the model
    predictions = model.predict(X_test)
    mse = np.mean((y_test.flatten() - predictions) ** 2)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predicted Prices', color='orange')
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Price')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    plt.show()

    report = f"Machine Learning Model Evaluation:\nMean Squared Error: {mse:.4f}\n"
    return report
