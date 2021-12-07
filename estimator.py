from math import sin
from pathlib import Path

import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
import time
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
import datetime





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
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't','raf10','ff','rr1']].sort_values('date').fillna(0), on='date')
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
    periods =[('month', 12),('day', 31),('weekday',7),('hour',24), ('week', 52)]
    #columns_drop = ['weekday','hour']
    #periods =[('weekday',7),('hour',24)]
    for element in periods:
        cos_col = element[0] + '_cos'
        sin_col = element[0] + '_sin'
        X.loc[:, cos_col] = X[element[0]].apply(lambda x: np.cos(x / element[1] * 2 * np.pi))
        X.loc[:, sin_col] = X[element[0]].apply(lambda x: np.sin(x / element[1] * 2 * np.pi))
    return X.drop(columns=columns_drop)


def categories():
    return [['Pont de Bercy SO-NE', '6 rue Julia Bartet NE-SO', '6 rue Julia Bartet SO-NE',
    'Totem Cours la Reine O-E', "Face au 25 quai de l'Oise NE-SO", "Face au 25 quai de l'Oise SO-NE",
    'Totem Cours la Reine E-O', 'Totem 73 boulevard de Sébastopol N-S', '152 boulevard du Montparnasse E-O',
    "Totem 85 quai d'Austerlitz SE-NO", 'Totem 64 Rue de Rivoli E-O', 'Totem 64 Rue de Rivoli O-E',
    "Totem 85 quai d'Austerlitz NO-SE", 'Pont des Invalides S-N', 'Pont de la Concorde S-N', '152 boulevard du Montparnasse O-E',
    'Face au 70 quai de Bercy S-N', 'Face au 70 quai de Bercy N-S', "Face 104 rue d'Aubervilliers S-N",
    '39 quai François Mauriac NO-SE', '39 quai François Mauriac SE-NO', "18 quai de l'Hôtel de Ville NO-SE",
    '20 Avenue de Clichy SE-NO', "18 quai de l'Hôtel de Ville SE-NO", 'Voie Georges Pompidou NE-SO',
    '20 Avenue de Clichy NO-SE', 'Voie Georges Pompidou SO-NE', '67 boulevard Voltaire SE-NO', '90 Rue De Sèvres SO-NE',
    'Face au 48 quai de la marne NE-SO', 'Face au 48 quai de la marne SO-NE', '90 Rue De Sèvres NE-SO',
    "Face 104 rue d'Aubervilliers N-S", 'Totem 73 boulevard de Sébastopol S-N', '27 quai de la Tournelle SE-NO',
    '28 boulevard Diderot E-O', '28 boulevard Diderot O-E', "Quai d'Orsay O-E", 'Pont Charles De Gaulle SO-NE',
    '38 rue Turbigo SO-NE', 'Pont des Invalides N-S', "Quai d'Orsay E-O", '36 quai de Grenelle SO-NE',
    'Pont Charles De Gaulle NE-SO', 'Face au 4 avenue de la porte de Bagnolet O-E', "Face au 40 quai D'Issy NE-SO",
    'Face au 4 avenue de la porte de Bagnolet E-O', '38 rue Turbigo NE-SO', '36 quai de Grenelle NE-SO',
    "Face au 40 quai D'Issy SO-NE", 'Pont de Bercy NE-SO', '27 quai de la Tournelle NO-SE',
    'Face au 8 avenue de la porte de Charenton NO-SE', 'Face au 8 avenue de la porte de Charenton SE-NO',
    '254 rue de Vaugirard SO-NE', '254 rue de Vaugirard NE-SO']]

def _lockdown(X):
    X = X.copy()
    lockdown1 = pd.date_range(start='03/17/2020',end='05/10/2020')
    lockdown2 = pd.date_range('10/30/2020','12/15/2020')
    lockdown3 = pd.date_range('4/3/2021','5/3/2021')
    def in_lockdown(x):
        if x in lockdown1: return 1
        elif x in lockdown2: return 2
        elif x in lockdown3: return 3
        else: return 0
    total_dates = pd.date_range(X.date.dt.date.min(),X.date.dt.date.max())
    scores = np.zeros(len(total_dates))
    final_serie = pd.DataFrame(data=scores, index = total_dates,columns=['lockdown']).reset_index()
    final_serie.loc[:,'lockdown']=final_serie['index'].apply(in_lockdown).values
    final_serie.rename(columns={'index':'Date'},inplace=True)
    final_serie.loc[:,'Date'] =pd.to_datetime(final_serie['Date'])
    X['Date'] = pd.to_datetime(X['date'].dt.date)
    return X.merge(final_serie,left_on='Date',right_on='Date').drop(columns=['Date'])

def _curfew(X):
    X = X.copy()
    curfew1 = pd.date_range(start='12/15/2020',end='01/16/2021',freq='1H') #20h - 6h
    curfew2 = pd.date_range('01/16/2021','03/20/2021',freq='1H') #18h - 6h
    curfew3 = pd.date_range('03/20/2021','05/19/2021',freq='1H') #19h - 6h
    curfew4 = pd.date_range('05/19/2021','06/09/2021',freq='1H') #21h - 6h
    
    def in_curfew(x):
        if x in curfew1 and (x.hour>= 19 or x.hour<=6):
            return 1
        elif x in curfew2 and (x.hour>= 18 or x.hour<=6):
            return 1
        elif x in curfew3 and (x.hour>= 19 or x.hour<=6):
            return 1
        elif x in curfew4 and (x.hour>= 21 or x.hour<=6):
            return 1
        else:
            return 0
    total_dates = pd.date_range(X.date.dt.date.min(),X.date.dt.date.max() + datetime.timedelta(days=1), freq = "1H")
    scores = np.zeros(len(total_dates))
    final_serie = pd.DataFrame(data=scores, index = total_dates,columns=['curfew']).reset_index()
    final_serie.loc[:,'curfew']=final_serie['index'].apply(in_curfew).values
    final_serie.rename(columns={'index':'Date'},inplace=True)
    final_serie.loc[:,'Date'] =pd.to_datetime(final_serie['Date'])
    X['Date'] = pd.to_datetime(X['date'])
    return X.merge(final_serie,left_on='Date',right_on='Date').drop(columns=['Date'])

def get_estimator():

    date_encoder = FunctionTransformer(_encode_dates, validate=False)
    computing_center = FunctionTransformer(_distance_center_paris, validate=False)
    columns_dropper = FunctionTransformer(_dropping_columns, validate=False)
    curfew_encoder = FunctionTransformer(_curfew, validate=False)
    lockdown_encoder = FunctionTransformer(_lockdown, validate=False)
    sinusoidale_dates = FunctionTransformer(_sinusoidale_dates,validate=False)
    external_data_merger = FunctionTransformer(_merge_external_data, validate=False)
    schools_holidays = FunctionTransformer(_schools_holidays,validate=False)
    
    categorical_cols = ["counter_name"]#,'site_name']
    transformed_dates = []
    for element in  ['month','day','weekday','hour', 'week'] :
        transformed_dates.append(element+'_cos')
        transformed_dates.append(element+'_sin')
    date_cols = ['year']
    
    deleted_cols = ['counter_id', 'site_id', 'counter_installation_date',
        'counter_technical_id']
    merging_cols = ['site_name']
    numeric_cols = ['latitude','longitude']+['t','ff']+['curfew','lockdown']

    preprocessor = ColumnTransformer([
        #('date', "passthrough", date_cols),
        ('cat', OneHotEncoder(categories=categories()), categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
        ('transformed_dates', 'passthrough', transformed_dates)
    ])
    regressor = LGBMRegressor(max_depth=14, num_leaves=50)#Ridge(alpha=0.0001)#RandomForestRegressor(n_estimators=80, max_depth=9)

    pipe =  Pipeline([
        #("FeatureCurfew",  curfew_encoder),
        #("FeatureLockdown",  lockdown_encoder),
        #("FeatureHolidays", schools_holidays),
        ("featureTemperature", external_data_merger),
        ("get_dates", date_encoder),
        ("sinusoidales_dates",sinusoidale_dates),
        ("drop_useless_cols", columns_dropper),
        ("prepocessor", preprocessor),
        ("model", LGBMRegressor(num_leaves=50))])

    return pipe





