from os.path import join
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import tsfel
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from skopt import BayesSearchCV

import utils
import metrics
from add_features import add_time_lags, add_calendar_features, add_aggregations
from plots import plot_actuals_vs_predictions

def bayesian_hyperparam_search(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> lgb.LGBMRegressor:
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    param_space = {
        'num_leaves': (20, 50, 70),
        'learning_rate': (0.01, 0.1, 'log-uniform'),
        'feature_fraction': (0.7, 1.0, 'uniform'),
        'bagging_fraction': (0.7, 1.0, 'uniform'),
        'max_depth': (3, 10, 15),
        'min_child_samples': (5, 30)
    }

    # Set up bayesian optimisation
    opt = BayesSearchCV(
        estimator=lgb.LGBMRegressor(objective='regression', metric='mean_squared_error', boosting_type='gbdt'),
        search_spaces=param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )

    opt.fit(X_train, y_train)

    best_estimator = opt.best_estimator_

    y_pred = best_estimator.predict(X_test, num_iteration=best_estimator.best_iteration)

    mse = mean_squared_error(y_test, y_pred)

    print(f"Best parameters found: {opt.best_params_}")
    print(f"Mean Squared Error on test set: {mse}")

    return opt.best_params_

def _perform_algorithmic_partitioning(df: pd.DataFrame) -> tuple:
    '''Instead of training a global forecasting model on the entire dataset,
    partitioning it into smaller almost equally sized parts has been shown
    to improve performance. I could perform this partition on essentially
    any feature or combination of features (for example, grouping by countries
    whose time series have been shown to be strongly correlated), but a better
    solution is to perform a clustering approach such as k-means to algorithmically
    find the time series which are most similar.

    To effictively cluster time series we first have to extract richer time series
    characteristics (autocorrelation, mean, variance, peak-to-peak distance, entropy
    etc.) to allow the model to compare the distances between them. This can be done
    automatically with tsfel (time series feature extraction library).

    Charu C. Agarwal showed in 2001 that euclidian distance and other common distance
    metrics are often not effective in high-dimensional space, so instead of performing
    k-means directly on our tsfel feature dataset (which contains dozens of features),
    we should first reduce the dimensionality and perform k-means on the
    lower-dimensional space.

    A good way to do this is to use t-SNE, which projects the higher dimensional
    points into a lower dimensional space whilst trtying to maintain the distribution
    of distance between each point in the original space.

    Once we have a few clusters, we can separate the dataset and train a gfm for each
    of them, which should improve performance.
    '''
    
    cfg = tsfel.get_features_by_domain()
    features = tsfel.time_series_features_extractor(cfg, df, fs=None)

    print(features)

    # Compute pairwise distances in the original space
    distances_original = pairwise_distances(X, metric='euclidean')
    prob_original = np.exp(-distances_original ** 2 / (2. * np.var(distances_original)))

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # Compute pairwise distances in the new space
    distances_new = pairwise_distances(X_embedded, metric='euclidean')
    prob_new = 1 / (1 + distances_new ** 2)

    # Normalize the probabilities
    prob_original /= np.sum(prob_original)
    prob_new /= np.sum(prob_new)

    # Plot the distributions
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(prob_original.ravel(), bins=50, color='blue', alpha=0.7)
    plt.title('Original Space Probability Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(prob_new.ravel(), bins=50, color='green', alpha=0.7)
    plt.title('New Space Probability Distribution')

    plt.show()

    print(df_embedded)

    plt.figure(figsize=(10, 7))
    plt.scatter(df_embedded[:, 0], df_embedded[:, 1], c='blue', marker='o')
    plt.title('t-SNE visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def train_lgbm_global(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    save_as_filename: str
) -> None:
    '''Instead of training a local model for each individual time series,
    it is often more effective and easier to train and maintain a one or
    several global models for the entire dataset (of all stores' time series).
    This provides the model with much more data, allows cross-learning
    across multiple time series, reduces the overall generalisation error
    across all forecasts, and allows a more complex model to be trained
    without overfitting, which includes adding many more lag features than
    would be possible with either traditional arima-type models or local
    ai forecasting models.
    '''
    
    # Find best hyperparams using bayesian optimisation
    #best_params = bayesian_hyperparam_search(
    #    X_train.values, y_train.values, X_test.values, y_test.values
    #)

    #print(best_params)
    
    # Retrain with the best params for more iterations
    best_params = {
        'boosting_type': 'gbdt',
        'bagging_fraction': 0.7629144166677786,
        'feature_fraction': 1.0,
        'learning_rate': 0.1,
        'max_depth': 15,
        'min_child_samples': 25,
        'num_leaves': 70,
        'reg_alpha': 0.1  # L1 regularization for embedded feature selection
    }

    num_iters = 1000

    # Convert dataset for lgb
    train_data = lgb.Dataset(X_train.values, label=y_train.values)
    test_data = lgb.Dataset(X_test.values, label=y_test.values, reference=train_data)

    estimator = lgb.train(
        best_params,
        train_data,
        num_iters,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=3),
        ])
    
    # Save model object
    utils.save_pickle(estimator, join('models', save_as_filename))

    y_pred: np.ndarray = estimator.predict(X_test, num_iteration=estimator.best_iteration)

    # Convert predictions to a pandas Series with the same index as y_test
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Calculate residuals
    residuals = y_pred - y_test

    # Also calculate in-sample predictions for metric calculation
    y_train_pred = estimator.predict(X_train, num_iteration=estimator.best_iteration)

    # Display performance metrics and charts
    train_mse = metrics.mse(y_train_pred, y_train.values)
    test_mse = metrics.mse(y_pred, y_test.values)

    bias = metrics.bias(y_train_pred, y_train)
    variance = metrics.variance(y_pred)

    print(f'In-sample MSE: {train_mse:.6f}. Out-of-sample MSE: {test_mse:.4f}')
    print(f'Bias: {bias:.6f}. Variance: {variance:.4f}')

def test_gfm(X_test: pd.Series, y_test: pd.Series, model_filename: str):

    estimator = utils.load_pickle(join('models', model_filename))

    y_pred: np.ndarray = estimator.predict(X_test, num_iteration=estimator.best_iteration)
    y_pred = pd.Series(y_pred, index=y_test.index)

    plot_actuals_vs_predictions(y_pred, y_test, last_n_days=7)

if __name__ == '__main__':

    model_name = 'lightgbm_day_ahead'

    # Use k-means to find optimal gfm partitions
    df_raw = pd.read_csv(join('data', 'EMHIRESPV_TSh_CF_Country_19862015.csv'))
    _perform_algorithmic_partitioning(df_raw)
    
    '''df = pd.read_csv(join('data', 'all_countries_cleaned_30_years.csv'))
    df['Date'] = pd.to_datetime(df['Date'])     
    
    # Add features

    # More complex gfms typically benefit from adding more lags, but often
    # overfits more if num lags is equal to a strong seasonal trend length
    df = add_time_lags(df, target_key='Wind_Energy_Potential', lags=48)
    df = add_calendar_features(df, datetime_feature='Date', num_fourier_terms=2)
    df = add_aggregations(df)

    # Remove Date feature before creating training and test set
    df = df.drop('Date', axis=1)
    
    # Split up the dataset
    X = df.drop(columns=['Wind_Energy_Potential'])
    y = df['Wind_Energy_Potential']

    # Train GFM---------------------------------------------------------------------------
    
    X_train, X_test, y_train, y_test = _train_test_split(raw_dataset=df, train_fraction=0.85)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    train_gfm(X_train, X_test, y_train, y_test, save_as_filename=f'{model_name}.pkl')

    # TODO: Remove most unimportant features and re-train
    
    # Test GFM for single country----------------------------------------------------------
    _, X_test_fr, _, y_test_fr = _train_test_split(raw_dataset=df, train_fraction=0.95, country_code='PL')
    
    # Add datetime index back. TODO: Make this cleaner
    t = pd.date_range('1/1/1986', periods = len(X_test_fr.index), freq = 'h')
    X_test_fr.index = t
    y_test_fr.index = t
    
    test_gfm(X_test_fr, y_test_fr, f'{model_name}.pkl')'''