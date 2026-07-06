import numpy as np
import joblib
from sklearn.linear_model import LinearRegression


def train_model():
    """Train a simple linear regression model"""
    print("Training model...")

    # Sample training data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save model to disk
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl")
    print(f"Model coefficient: {model.coef_[0]:.2f}")
    print(f"Model intercept: {model.intercept_:.2f}")


if __name__ == '__main__':
    train_model()
