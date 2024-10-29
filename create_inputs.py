from os.path import join
import pandas as pd
from os.path import join
from add_features import add_calendar_features, add_time_lags, add_aggregations

def create_inputs(family: str) -> None:

    df_train = pd.read_csv(join('data', 'train.csv'))
    df_oil = pd.read_csv(join('data', 'oil.csv')).set_index('date')
    df_transactions = pd.read_csv(join('data', 'transactions.csv')).set_index('date')
    df_stores = pd.read_csv(join('data', 'stores.csv'))

    df_train['date'] = pd.to_datetime(df_train['date'])
    df_train = df_train.merge(df_stores[['store_nbr', 'cluster']], on='store_nbr', how='left')

    df_oil.index = pd.to_datetime(df_oil.index)
    #df_transactions.index = pd.to_datetime(df_transactions.index)

    # Interpolate the missing oil price data (simple linear interpolation for now). See eda.ipynb but since
    # oil price appears to have a more macro, step-impact on magazine sales I think this should be okay.
    df_oil = df_oil.interpolate(method='linear')

    df = pd.merge(df_train, df_oil, how='left', left_on='date', right_on='date')
    #df = pd.merge(df, df_transactions, how='left', on=['date', 'store_nbr'])

    #df = df.rename(columns={'transactions': 'daily_store_transactions'})

    df_family = df[df['family'] == family]

    # Add some derived temporal features
    df_family = add_calendar_features(df_family, datetime_feature='date', num_fourier_terms=2)
    
    # GFM will be more complex model that benefit from more lags. 50 days to avoid possible monthly
    # seasonal trend, which can sometimes lead to overfitting
    df_family = add_time_lags(df_family, target_key='sales', lags=50)
    
    # Weekly, monthly, daily rolling means and sums + ewma
    df_family = add_aggregations(df_family, target_key='sales')

    # Remove nan created by shifts
    df_family = df_family.dropna()

    # Drop non-numeric or redundant columns
    df_family = df_family.drop(columns=['id', 'family'])
    
    df_family.to_csv(join('data', f'{family.lower()}_train_cleaned.csv'))

def train_test_split(df: pd.DataFrame, target_feature: str, train_fraction: float, time_series_categorical: str, time_series_ids: list[int]) -> tuple[pd.Series]:

    # If more than one country, split the data so that each country is equally represented
    # in both training and testing sets
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for id in time_series_ids:
        country_data = df[df[time_series_categorical] == id]
        train_size = int(len(country_data) * train_fraction)
        
        # Append to full datasets
        train_data = pd.concat([train_data, country_data.iloc[:train_size]])
        test_data = pd.concat([test_data, country_data.iloc[train_size:]])
    
    X_train = train_data.drop(columns=[target_feature])
    y_train = train_data[target_feature]

    X_test = test_data.drop(columns=[target_feature])
    y_test = test_data[target_feature]
    
    X_train.astype(float)
    X_test.astype(float)
    y_train.astype(float)
    y_test.astype(float)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    create_inputs(family='BEVERAGES')