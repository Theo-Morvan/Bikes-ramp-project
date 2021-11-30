from math import sin
from pathlib import Path

import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns
from xgboost import XGBRegressor





sns.set_style('darkgrid')




def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour
    X.loc[: , 'week'] = X['date'].dt.week
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"]) 

def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't','ff']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X

def _distance_center_paris(X):
    X = X.copy()
    geoloc = X.drop_duplicates(['site_name'])[['site_name','latitude','longitude']]
    df_append = pd.DataFrame(data = [['center']+list(geoloc.drop_duplicates(['site_name'])[['latitude','longitude']].mean().values)]
                             , columns=['site_name','latitude','longitude'])
    geoloc =geoloc.append(df_append)
    geoloc['distance_center'] = (geoloc.iloc[:,1:] - geoloc.iloc[-1,1:]).apply(np.linalg.norm,axis=1)
    X = X.merge(geoloc.iloc[:-1,[0,-1]],left_on='site_name',right_on='site_name')
    return X

def _dropping_columns(X):
    X = X.copy()

    return  X.drop(columns = {"site_name",'counter_id', 'site_id', 'counter_installation_date','counter_technical_id'})

def _schools_holidays(X):
    X = X.copy()
    toussaint = pd.date_range(start='10/17/2020',end='11/2/2020')
    xmas = pd.date_range('12/19/2020','1/4/2021')
    winter = pd.date_range('12/19/2020','1/3/2021')
    paques = pd.date_range('4/10/2021','4/26/2021')
    summer = pd.date_range('6/7/2021','9/4/2021')
    holidays = list(toussaint)+list(xmas)+list(winter)+list(paques)
    total_dates = pd.date_range(X.date.dt.date.min(),X.date.dt.date.max())
    scores = np.zeros(len(total_dates))
    final_serie = pd.DataFrame(data=scores, index = total_dates,columns=['Holiday']).reset_index()
    final_serie.loc[:,'Holiday']=final_serie['index'].apply(lambda date : 1 if date in holidays else 0).values
    final_serie.rename(columns={'index':'Date'},inplace=True)
    final_serie.loc[:,'Date'] =pd.to_datetime(final_serie['Date'])
    X['Date'] = pd.to_datetime(X['date'].dt.date)
    return X.merge(final_serie,left_on='Date',right_on='Date').drop(columns=['Date'])

def _sinusoidale_dates(X):
    X = X.copy()  # modify a copy of X
    columns_drop = ['month','day','weekday','hour', 'week']
    periods =[('month', 12),('day',31),('weekday',7),('hour',24), ('week', 52)]
    for element in periods:
        cos_col = element[0] + '_cos'
        sin_col = element[0] + '_sin'
        X.loc[:, cos_col] = X[element[0]].apply(lambda x: np.cos(x / element[1] * 2 * np.pi))
        X.loc[:, sin_col] = X[element[0]].apply(lambda x: np.sin(x / element[1] * 2 * np.pi))
    return X.drop(columns=columns_drop)




def get_estimator():
    #data = pd.read_parquet(Path('data') / 'train.parquet')
    #data['log_bike_count'] = np.log(1 + data['bike_count'])
    date_encoder = FunctionTransformer(_encode_dates, validate=False)
    date_encoder = FunctionTransformer(_encode_dates)
    computing_center = FunctionTransformer(_distance_center_paris, validate=False)
    columns_dropper = FunctionTransformer(_dropping_columns, validate=False)
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ['year', 'month','day','weekday','hour','week']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name"]#,'site_name']

    transformed_dates = []
    for element in  ['month','day','weekday','hour', 'week'] :
        transformed_dates.append(element+'_cos')
        transformed_dates.append(element+'_sin')
    sinusoidale_dates = FunctionTransformer(_sinusoidale_dates,validate=False)
    deleted_cols = ['counter_id', 'site_id', 'counter_installation_date',
        'counter_technical_id']
    merging_cols = ['site_name']
    numeric_cols = ['latitude','longitude']+['t','ff','distance_center']  + ['Holiday','year'] #+ transformed_dates
    preprocessor = ColumnTransformer([
        ('date', "passthrough", date_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('numeric', 'passthrough', numeric_cols),
    ])
    regressor = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.001)
    pipe =  make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        FunctionTransformer(_schools_holidays,validate=False),
        date_encoder,
        computing_center,
        #sinusoidale_dates,
        columns_dropper,
        preprocessor,
        regressor
    )
    return pipe





