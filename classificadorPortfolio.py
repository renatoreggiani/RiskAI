# -*- coding: utf-8 -*-
"""ClassificadorACWI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aKvEZhXxFSdh_iCN7wwGWV1hfBcT7Vmj

# Importando bibliotecas
"""



import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

from preprocess import get_datas

# Necessario criar a key no quandl

#%%
def get_portfolio(indice='portfolio'):
    if indice=='portfolio':
        df = pd.read_csv('dados/retorno_portfolio.csv', parse_dates=['Date'])
        df.fillna(0, inplace=True)
    if indice=='ACWI':    
        df = pd.read_csv('dados/retorno_portfolio.csv', parse_dates=['Date'])
        df.fillna(0, inplace=True)
        df['retorno'] = df['ACWI']
        
    df['year_week'] = df['Date'].dt.strftime('%Y-%U')
    df['retorno acumulado'] = df['retorno'] + 1
    df['retorno acumulado'] = df['retorno acumulado'].cumprod()
    
    df_gsrai = pd.read_csv('dados/GSRAII.csv', parse_dates=['Date']).sort_values('Date')

    df = df.merge(df_gsrai, on='Date')
    return df[['Date', 'retorno acumulado', 'GSRAII Index', 'year_week']].dropna()

def classe_gsrai(df, limit=0, media_movel=3):
    
    df=df.copy()
    df['mean'] = df['GSRAII Index'].rolling(media_movel).mean()
    up_down = df['mean']

    df['gsrai_gt_up'] = np.where((df['GSRAII Index'] > up_down) & (df['GSRAII Index'] >= limit), 1, 0)
    df['gsrai_gt_down'] = np.where((df['GSRAII Index'] < up_down) & (df['GSRAII Index'] >= limit), 1, 0)
    df['gsrai_lt_up'] = np.where((df['GSRAII Index'] > up_down) & (df['GSRAII Index'] < limit), 1, 0)
    df['gsrai_lt_down'] = np.where((df['GSRAII Index'] < up_down) & (df['GSRAII Index'] < limit), 1, 0)

    regras = ['gsrai_gt_up', 'gsrai_gt_down', 'gsrai_lt_up', 'gsrai_lt_down']
    df['classe'] = np.nan
    for col in regras:
        df.loc[df[col]==1, 'classe'] = col
    return df.dropna()

#%% Semanal
df = get_portfolio()
df_gp = df.groupby('year_week').agg('last')

# for pct in range(1,6):
#     df_gp[f'pct_{pct}_sem'] = (df_gp['retorno acumulado']/df_gp['retorno acumulado'].shift(pct).values -1) * 100
   
    
for pct in range(1,6):
    df_gp[f'pct_{pct}_sem'] = (df_gp['retorno acumulado'].shift(pct*-1).values/
                               df_gp['retorno acumulado'] -1) * 100
    
    
df_gp =  classe_gsrai(df=df_gp, limit=0, media_movel=3)
    
df = get_datas(return_df=True)


df_all = df_gp.merge(df, right_index=True, left_index=True)

# 'pct_1_sem', 'pct_2_sem', 'pct_3_sem', 'pct_4_sem', 'pct_5_sem',
pct = 'pct_3_sem'
mean = df_all[df_all[pct]<0][pct].mean()
df_all['category'] = np.where(df_all[pct] < mean, 1, 0)
df_all['category'] = df_all['category'].shift(-1).values

v = df_all[['category', 'retorno acumulado', 'pct_1_sem', 'pct_2_sem',
            'pct_3_sem', 'pct_4_sem', 'pct_5_sem']]

#%%

df_model = df_all[[pct, 'GSRAII Index_x',  'category',
               'gsrai_gt_up', 'gsrai_gt_down', 'gsrai_lt_up', 'gsrai_lt_down', 
               'FEDFUNDS', 'CPALTT01USM657N', 'VIXCLS', 'DGS10',
               'AAA10Y', 'BAMLH0A0HYM2EY', 'GEPUCURRENT', 'DGS3MO', 'acwi_log_diff',
               'pmi_us_gt_50_up', 'pmi_us_gt_50_down', 'pmi_us_lt_50_up',
               'pmi_us_lt_50_down', 'BAMLEMCBPIOAS', 'USTREASURY/HQMYC: 10.0-20.0',
               'USTREASURY/YIELD: 10 YR-20 YR']].dropna()


y = df_model['category'].values
X = df_model.drop(columns=['category']).values
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)


#%% Funções e seed para os modelos

class_weight = {1: y[y == 0].size / y.size,
                0: y[y == 1].size / y.size} 

SEED = 51
N_ITER =  50   
N_SPLITS = 50
ss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=SEED)

def valid(model, X, y):
  ss_valid = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=666)
  scores = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=ss_valid, n_jobs=-1)
  results = {
      'Modelo': str(model).split('(')[0],
      'Media': scores.mean(),
      'std': scores.std(),
      'Pior': scores.min(),
      'Melhor': scores.max(),
      'Parametros': model.get_params()
  }
  return pd.DataFrame([results])

  
def hp_tunning(model, params, random_state=SEED, n_iter=N_ITER, cv=ss):
  print(f'Testando hiperparametros para {str(model).split("(")[0]}' )
  clf = RandomizedSearchCV(model, params, random_state=random_state, scoring='balanced_accuracy',
                           n_iter=n_iter, cv=cv, n_jobs=-1, verbose=1)
  rsearch = clf.fit(X, y)
  df_rs = pd.DataFrame(rsearch.cv_results_)
  df_rs = df_rs[[col for col in df_rs.columns if not col.startswith('split')]].sort_values('rank_test_score')
  return rsearch.best_params_, df_rs



#%% SGDClassifier

sgd = SGDClassifier(random_state=SEED)
df_default_sgd = valid(sgd, X, y)
df_default_sgd


### Ajuste de Hiperparametros

params_sgd = {
    'loss':['hinge', 'log', 'modified_huber'],
    'penalty':['l2', 'l1', 'elasticnet'],
    'alpha':[1e-4, 1e-3, 1e-2],
    'max_iter':range(15000, 19001, 1000), 
    'n_iter_no_change': range(10, 21, 2),
    'class_weight':[class_weight]
    }

sgd = SGDClassifier(random_state=SEED)
best_params_sgd, df_rs_sgd = hp_tunning(sgd, params_sgd)

df_rs_sgd.head()

### Melhor SGDClassifier


best_params_sgd 
# {'alpha': 0.001, 'average': False, 'class_weight': {1: 0.7574698333652558, 0: 0.2425301666347443}, 'early_stopping': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': 6000, 'n_iter_no_change': 16, 'n_jobs': None, 'penalty': 'l2', 'power_t': 0.5, 'random_state': None, 'shuffle': True, 'tol': 0.001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}

best_sgd = SGDClassifier(**best_params_sgd)
df_best_sgd = valid(best_sgd, X, y)
df_best_sgd


#%% SVC

svc = SVC(random_state=SEED)
df_default_svc = valid(svc, X, y)
df_default_svc

### Ajuste de hiperparametros

params_svc = {
    'C': range(10, 201, 10),
    'kernel':['poly', 'sigmoid', 'rbf', 'linear'],
    'gamma':['scale', 'auto'],
    'degree': range(1, 8, 2),
    'max_iter':[20000],
    'class_weight':[class_weight]
    }

svc = SVC(random_state=SEED)
best_params_svc, df_rs_svc = hp_tunning(svc, params_svc)

df_rs_svc.head()

### Melhor SVC



best_params_svc  
# {'C': 40, 'break_ties': False, 'cache_size': 200, 'class_weight': {1: 0.7574698333652558, 0: 0.2425301666347443}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 7, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': 10000, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}


df_best_svc = valid(SVC(**best_params_svc), X,y)
df_best_svc


df_default_svc



#%% Random forest


rfc = RandomForestClassifier(random_state=SEED)
df_default_rfc = valid(rfc, X, y)
df_default_rfc

### Ajuste de hiperparametros

params_rfc = {
    'n_estimators':range(80, 161, 10),
    'criterion':['gini', 'entropy'], 
    'max_depth':range(20, 60, 3), 
    'min_samples_split':range(20, 41, 2), 
    'min_samples_leaf':range(45, 55, 2), 
    'max_features':['log2', 'auto'],
    'class_weight':[class_weight]
    }

rfc = RandomForestClassifier(random_state=SEED)
best_params_rfc, df_rs_rfc = hp_tunning(rfc, params_rfc)

df_rs_rfc.head()

### Melhor RFC



best_params_rfc 
# {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': {1: 0.7574698333652558, 0: 0.2425301666347443}, 'criterion': 'entropy', 'max_depth': 23, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 51, 'min_samples_split': 24, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 120, 'n_jobs': None, 'oob_score': False, 'random_state': 51, 'verbose': 0, 'warm_start': False}



rfc = RandomForestClassifier(**best_params_rfc, random_state=SEED)
df_best_rfc = valid(rfc, X, y)
df_best_rfc

rfc.fit(X, y)

df_default_rfc

#%% Logistic Regression



logr = LogisticRegression()
df_default_logr = valid(logr, X, y)

### Ajuste de hiperparametros
    # 'newton-cg' - ['l2', 'none']
    # 'lbfgs' - ['l2', 'none']
    # 'liblinear' - ['l1', 'l2']
    # 'sag' - ['l2', 'none']
    # 'saga' - ['elasticnet', 'l1', 'l2', 'none']

params_logr = {
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'penalty':['l2', 'none'],
    'C': range(1, 201, 10),
    'max_iter':range(500, 2001, 100), 
    'class_weight':[class_weight]
    }


logr = LogisticRegression(random_state=SEED)
best_params_logr, df_rs_logr = hp_tunning(logr, params_logr)

df_rs_logr.head()

### Melhor LOGR


best_params_logr 


logr = LogisticRegression(**best_params_logr, random_state=SEED)
df_best_logr = valid(logr, X, y)
df_best_logr

logr.fit(X, y)

df_default_logr