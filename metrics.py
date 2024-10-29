import numpy as np

def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return np.mean((target - prediction)**2)

def bias(y_pred: np.ndarray, y_train: np.ndarray) -> float:
    
    return np.mean(y_train - y_pred)

def variance(y_pred: np.ndarray) -> float:
    y_pred_hat = np.mean(y_pred)

    return np.mean((y_pred - y_pred_hat)**2)

