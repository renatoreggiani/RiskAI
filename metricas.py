# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 20:35:28 2021

@author: F8564619
"""


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

#%%

qtd_amostras = 100

df = pd.DataFrame(
    {'real': np.random.randint(2, size=qtd_amostras),
     'previsto': np.random.randint(2, size=qtd_amostras),
     'previsto1': np.random.randint(2, size=qtd_amostras),
     'previsto2': np.random.randint(2, size=qtd_amostras),
     'probabilidade': np.random.random_sample(qtd_amostras)
     })

#%%

def validador(df, nome_modelo=None):
    df = df.copy()
    list_pct_conf = [0.6, 0.7, 0.75,  0.8, 0.85, 0.9]
    df_result = pd.DataFrame()

    for pct_conf in list_pct_conf:

        df_loop = df.copy()
        df_loop.iloc[:,1] = np.where((df_loop.iloc[:,1]==1) & (df_loop.iloc[:,2]>=pct_conf), 1, 0)

        tn, fp, fn, tp = confusion_matrix(df_loop.iloc[:,0], df_loop.iloc[:,1]).ravel()

        metricas = {
            'pct_conf': pct_conf,
            'f1': [f1_score(df_loop.iloc[:,0], df_loop.iloc[:,1])]
            }
        metricas['acuracia'] = (tp + tn) / (len(df_loop))
        metricas['precisao'] = tp / (tp + fp)
        metricas['recall'] = tp / (tp + fn)
        metricas['v_neg'], metricas['f_pos'], metricas['f_neg'], metricas['v_pos'] = tn, fp, fn, tp
        df_result = df_result.append(pd.DataFrame(metricas))

    if nome_modelo:
        df_result['modelo'] = nome_modelo

    return df_result



df_val = validador(df)


