'''
Ideas for Project:

1. Train and tune a multi-step mlp gfm to create point predictions for magazine sales
the next five days (Easier since exogenous input variables make a purely recursive
approach less simple to implement for multi-step prediction).

2. Rather than only predicting expected value, create a probabilistic forecast that
tries to predict the probability density function (predict mean and standard deviation)

3. Create and display various performance metrics to assess how well the model is
doing and take the necessary steps to improve the performance.

4. Try and create an ensemble of a few different regressors (e.g. fully connected, gbm and
maybe an lstm) and either take a weighted combination of their outputs or train a small meta
model (e.g. linear regression) using their out of sample predictions as inputs to see if
this improves the performance any more.

'''

from os.path import join
import pandas as pd
from create_inputs import train_test_split
from lgbm import train_lgbm_global

if __name__ == '__main__':
    
    model_name = 'lightgbm_day_ahead'
    datetime_feature = 'date'
    target_feature = 'sales'

    df = pd.read_csv(join('data', 'beverages_train_cleaned.csv'))

    #df[datetime_feature] = pd.to_datetime(df[datetime_feature])     
    df = df.drop(datetime_feature, axis=1)
    
    # Split up the dataset
    X = df.drop(columns=[target_feature])
    y = df[target_feature]

    # Train GFM---------------------------------------------------------------------------
    
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target_feature,
        train_fraction=0.85,
        time_series_categorical='store_nbr',
        time_series_ids=df['store_nbr'].unique()
        )

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    train_lgbm_global(X_train, X_test, y_train, y_test, save_as_filename=f'{model_name}.pkl')

    # TODO: Remove most unimportant features and re-train
    
    # Test GFM for single country----------------------------------------------------------
    #_, X_test_fr, _, y_test_fr = _train_test_split(raw_dataset=df, train_fraction=0.95, country_code='PL')
    
    # Add datetime index back. TODO: Make this cleaner
    #t = pd.date_range('1/1/1986', periods = len(X_test_fr.index), freq = 'h')
    #X_test_fr.index = t
    #y_test_fr.index = t
    
    #test_gfm(X_test_fr, y_test_fr, f'{model_name}.pkl')'''