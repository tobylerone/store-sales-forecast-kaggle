import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, datetime_feature: str, num_fourier_terms: int = 2) -> pd.DataFrame:
    """
    Adds discrete calendar features (ordinal step functions) and continuous Fourier terms to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        datetime_feature: Feature to use for time series information. Defaults to index
        num_fourier_terms (int): Number of Fourier terms to add (default is 5).
    
    Returns:
        pd.DataFrame: DataFrame with added features.

    Example:
        If num_fourier_terms = 2, this will create sine and cosine wave features with
        wavelengths corresponding to one full cycle per day (i=1), and two full cycles
        per day (i=2).
    """

    datetime_series = df[datetime_feature].dt if datetime_feature else df.index
    
    # Add discrete calendar features (Essentially an ordinal step function)
    df['year'] = datetime_series.year
    df['month'] = datetime_series.month
    df['week'] = datetime_series.isocalendar().week
    df['day_of_week'] = datetime_series.dayofweek

    # Add continuous calendar features (Continuous fourier terms).
    for i in range(1, num_fourier_terms + 1):
        
        # No hourly data
        #freq_daily = 2 * np.pi * i / 24  # Daily frequency (24 hours)
        #df[f'Hourly_Sin_{i}'] = np.sin(freq_daily * datetime_series.hour)
        #df[f'Hourly_Cos_{i}'] = np.cos(freq_daily * datetime_series.hour)

        freq_weekly = 2 * np.pi * i / 7
        df[f'daily_sin_{i}'] = np.sin(freq_weekly * datetime_series.day_of_week)
        df[f'daily_cos_{i}'] = np.cos(freq_weekly * datetime_series.day_of_week)

    return df

def add_time_lags(df: pd.DataFrame, target_key: str, lags: int = 48) -> pd.DataFrame:

    for i in range(1, lags + 1):
        df[f"lagged_{target_key}_{i}"] = df[target_key].shift(i)

    return df

def _add_ewma_features(df: pd.DataFrame, target_key: str, max_span: int = 3) -> pd.DataFrame:

    # Create contiguous copy to avoid fragmentation
    df = df.copy()
    
    # Shift to avoid leaking lag=0 into new features
    for span in range(2, max_span):
        df[f'{target_key}_ewma_span_{span}'] = df[target_key].ewm(span=span).mean().shift(1)

    return df

def add_aggregations(df: pd.DataFrame, target_key: str) -> pd.DataFrame:

    # Add rolling window aggregations

    # Add seasonal rolling window aggregations
    # Daily rolling mean
    df['weekly_rolling_mean'] = df[target_key].rolling(window=7).mean().shift(1)
    df['weekly_rolling_sum'] = df[target_key].rolling(window=7).sum().shift(1)

    # Monthly rolling sum
    df['monthly_rolling_mean'] = df[target_key].rolling(window=30).mean().shift(1)
    df['monthly_rolling_sum'] = df[target_key].rolling(window=30).sum().shift(1)

    # Yearly rolling sum
    #df['Yearly_Mean'] = df[target_key].rolling(window='365D').mean().shift(1)
    #df['Yearly_Sum'] = df[target_key].rolling(window='365D').mean().shift(1)

    # Add exponentially weighted moving averages
    df = _add_ewma_features(df, target_key, 60)

    return df