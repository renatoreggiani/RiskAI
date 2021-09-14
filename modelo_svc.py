# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:01:35 2021

@author: F8564619
"""


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from preprocess import get_datas


#%%

scaler = MinMaxScaler()

X, y = get_datas(scaler=scaler)

#%%

svc = SVC(C=10, degree=1)

SEED = 224

rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=SEED)
ss = ShuffleSplit(n_splits=100, test_size=0.25, random_state=SEED)

cv = ss

scores = cross_val_score(svc, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Media: ', scores.mean(),
      'Std: ', scores.std(),
      'Pior: ', scores.min(),
      'Melhor: ', scores.max(),
      sep='\n')


