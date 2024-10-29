import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_actuals_vs_predictions(y_pred: pd.Series, y_test: pd.Series, last_n_days: int = 7) -> None:
    
    print(type(y_test.index))
    
    # Only want the last 7 days
    end_date = y_test.index.max()
    start_date = end_date - pd.Timedelta(days=last_n_days)

    # Plot the target predictions against the actuals for the test set for the last month
    plt.plot(y_test[start_date:end_date], label='Actuals', color='blue')
    plt.plot(y_pred[start_date:end_date], label='Prediction', color='red')

    plt.xlabel('Time')
    plt.ylabel('Target')
    plt.title('Actuals vs Predictions')
    plt.legend()

    plt.show()

def plot_prediction_actuals_correlation(y_pred: np.ndarray, y_test: np.ndarray) -> None:

    plt.scatter(y_test, y_pred, color='blue', label='Test / Prediction Correlation')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    # Set axes to start at 0
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.title('Y_test / Y_pred Correlation')
    plt.legend()

def plot_residuals(residuals: pd.Series) -> None:
    # Plot the residuals
    # Plot the target predictions against the actuals for the test set for the last month
    plt.plot(residuals, label='Residuals', color='blue')

    plt.xlabel('Residuals')
    plt.ylabel('Time')
    plt.title('Residuals')
    plt.legend()

    plt.show()

