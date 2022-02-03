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
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from preprocess import get_datas


#%% Semanal

# df_gp =  classe_gsrai(df=df_gp, limit=0, media_movel=3)
diff_gsrai=-0.1

df = get_datas(return_df=True, diff_gsrai=diff_gsrai, periods=4)

scaler = MinMaxScaler()
X, y = get_datas(scaler=scaler, diff_gsrai=diff_gsrai, periods=4)

class_weight = {1: y[y == 0].size / y.size,
                0: y[y == 1].size / y.size}


#%% Funções e seed para os modelos

SEED = 51
N_ITER =  100
N_SPLITS = 25
ss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.25, random_state=SEED)


def valid(model, X, y, scoring='balanced_accuracy'):
    ss_valid = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=666)
    scores = cross_val_score(model, X, y, scoring=scoring, cv=ss_valid, n_jobs=-1)
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


def ft_selec_tunning(model, params, random_state=SEED, n_iter=N_ITER, cv=ss):

    param_grid = {"model__"+ k:v for k,v in params.items()}
    # param_grid = {}
    param_grid["pca__n_components"] = np.arange(5,24,1)

    # print(f'Testando hiperparametros para {str(model).split("(")[0]}' )
    pipe = Pipeline(steps=[("pca", PCA(svd_solver='full')), ("model", model)])
    clf = RandomizedSearchCV(pipe, param_grid, random_state=random_state, scoring='balanced_accuracy',
                             n_iter=n_iter, cv=cv, n_jobs=4, verbose=1)
    rsearch = clf.fit(X, y)
    df_rs = pd.DataFrame(rsearch.cv_results_)
    df_rs = df_rs[[col for col in df_rs.columns if not col.startswith('split')]].sort_values('rank_test_score')
    return rsearch.best_params_, df_rs, rsearch


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
    'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],
    # 'penalty':['l2'],
    'C': range(100, 301, 5),
    'max_iter':range(500, 2001, 100),
    'class_weight':[class_weight]
    }


logr = LogisticRegression(random_state=SEED)
best_params_logr, df_rs_logr = hp_tunning(logr, params_logr)
# sem lag 0.5466
# com lag 0.5954


# best_ft_params_logr, df_ft_logr, p = ft_selec_tunning(logr, params_logr)
# sem lag 0.5579
# com lag 0.5968


df_rs_logr.head()

### Melhor LOGR

best_params_logr

logr = LogisticRegression(**best_params_logr, random_state=SEED)
df_best_logr = valid(logr, X, y)


pipe = LogisticRegression(**best_params_logr, random_state=SEED)
df_best_logr = valid(logr, X, y)

logr.fit(X, y)

df_default_logr


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

svc = SVC(random_state=SEED, probability=True)
best_params_svc, df_rs_svc = hp_tunning(svc, params_svc)

df_rs_svc.head()

### Melhor SVC

best_params_svc

df_best_svc = valid(SVC(**best_params_svc, probability=True), X,y)
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



# Importancia das variaveis
import matplotlib.pyplot as plt
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)

forest_importances = pd.Series(importances, index=df.drop(columns='category').columns)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()

#%% MLPClassifier

from sklearn.neural_network import MLPClassifier




mlp = MLPClassifier(random_state=SEED)
df_default_mlp = valid(mlp, X, y)

params_mlp = {
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(15,), (20, 20), (15, 15, 30, 15), (5, 50)],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'max_iter':range(5000, 9001, 1000),
    }


mlp = MLPClassifier(random_state=SEED)
best_params_mlp, df_rs_mlp = hp_tunning(mlp, params_mlp)

df_rs_mlp.head()

### Melhor mlp


best_params_mlp


mlp = MLPClassifier(**best_params_mlp, random_state=SEED)
df_best_mlp = valid(mlp, X, y)
df_best_mlp

mlp.fit(X, y)

df_default_mlp

#%% GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = 66**2 * RBF(1.33)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0, multi_class='one_vs_one')
df_default_gcp = valid(gpc, X, y)



#%% AUTO ML


# import h2o
# from h2o.automl import H2OAutoML

# df_model.to_csv('dados_automl.csv', index=False)


# h2o.init()

# train = h2o.import_file("dados_automl.csv")
# # test = h2o.import_file("dados_automl.csv")

# # Identify predictors and response
# x = train.columns
# y = "category"
# # x.remove(y)

# # For binary classification, response should be a factor
# train[y] = train[y].asfactor()
# # test[y] = test[y].asfactor()

# # Run AutoML for 20 base models
# aml = H2OAutoML(max_models=50, seed=1,
#                 balance_classes=True,
#                 sort_metric='logloss'
#                 )
# aml.train(x=x, y=y, training_frame=train, )

# # View the AutoML Leaderboard
# lb = aml.leaderboard
# lb.head(rows=10)  # Print all rows instead of default (10 rows)

