# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:13:19 2021

@author: F8564619
"""

import pandas as pd
import numpy as np

from captura import get_acwi, get_fred, get_pmi_us_classified, get_quandl

#%% Captura dos dados

series_fred = ['FEDFUNDS', 'CPALTT01USM657N', 'VIXCLS', 'DGS10', 'AAA10Y',
               'BAMLH0A0HYM2EY', 'GEPUCURRENT',  'DGS3MO']

df = get_acwi()

df = df.merge(get_pmi_us_classified(), how='left', on='year_week').ffill()
df = df.merge(get_fred(series_fred), how='left', on='year_week').ffill()
df = df.merge(get_quandl("ML/EMCBI"), how='left', on='year_week').ffill()

# discutir sobre quais pontos pegar da curva abaixo
df = df.merge(get_quandl('USTREASURY/HQMYC', curve_diff=('10.0', '20.0')),
              how='left', on='year_week').ffill()

df = df.merge(get_quandl('USTREASURY/YIELD', curve_diff=('10 YR', '20 YR')),
              how='left', on='year_week').ffill()

df2 = get_quandl('USTREASURY/REALYIELD')['10 YR']

df.dropna(inplace=True)



#%%

df_gsrai = pd.read_csv('dados/GSRAII.csv', parse_dates=['Date']).sort_values('Date')
df_gsrai['year_week'] = df_gsrai['Date'].dt.strftime('%Y-%U')
df_gsrai = df_gsrai.groupby('year_week').agg('last')
df_gsrai.drop(columns='Date', inplace=True)


df_gsrai['GSRAII Index'].plot()

df = df.merge(df_gsrai, how='left', on='year_week').ffill()

# correlação desafada entre pontos da curva e difereça dos pontos
curvas = get_quandl('USTREASURY/HQMYC')



#%%

df_model = df.copy()

df_model['category'] = np.where(df_model['GSRAII Index'] < -0.01, 1, 0)
df_model['category'] = df_model['category'].shift(-1)
df_model.dropna(inplace=True)

X = df_model['category'].values
y = df_model.drop(columns='category').values

print( 'Shape df_model:', df_model.shape,
'Shape X:', X.shape,
'Shape y:',y.shape,
sep='\n')

df_model.plot()

# excluir PMI stardard scale
