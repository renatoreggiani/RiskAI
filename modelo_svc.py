# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:01:35 2021

@author: F8564619
"""
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from preprocess import get_datas

#%%

scaler = MinMaxScaler()

X, y = get_datas(scaler=scaler)

#%%

model = SVC()

SEED = 51

ss = ShuffleSplit(n_splits=100, test_size=0.25, random_state=SEED)

cv = ss

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Media: ', scores.mean(),
      'Std: ', scores.std(),
      'Pior: ', scores.min(),
      'Melhor: ', scores.max(),
      sep='\n')



#%%

params = {
    'C': range(1, 151, 10),
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma':['scale', 'auto'],
    # 'degree': range(1,10, 2),
    }

clf = RandomizedSearchCV(model, params, random_state=SEED, n_iter=50, cv=ss, n_jobs=-1)

rsearch = clf.fit(X, y)
rsearch.best_params_

df_rs = pd.DataFrame(rsearch.cv_results_)

df_rs = df_rs[[col for col in df_rs.columns if not col.startswith('split')]]


#%%

model = SVC(**rsearch.best_params_)

SEED = 666

ss = ShuffleSplit(n_splits=100, test_size=0.25, random_state=SEED)

cv = ss

scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Media: ', scores.mean(),
      'Std: ', scores.std(),
      'Pior: ', scores.min(),
      'Melhor: ', scores.max(),
      sep='\n')

