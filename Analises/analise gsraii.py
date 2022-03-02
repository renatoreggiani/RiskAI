# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:19:29 2021

@author: F8564619
"""
import pandas as pd
import numpy as np
import plotly.express as px


import plotly.graph_objects as go

import plotly.io as pio
#pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

#%%

df = pd.read_csv('dados/retorno_portfolio.csv', parse_dates=['Date'])
df.fillna(0, inplace=True)
df_gsrai = pd.read_csv('dados/GSRAII.csv', parse_dates=['Date']).sort_values('Date')

df = df.merge(df_gsrai, on='Date')
df['retorno'] = df['retorno']
df['retorno acumulado'] = df['retorno'] + 1
df['retorno acumulado'] = df['retorno acumulado'].cumprod()

df.drop(columns=['ACWI', 'BCOM Index', 'JPEGCOMP Index', 'SP500BDT Index', 'DGS10'], inplace=True)

df.sort_values('Date', inplace=True)
df['year_week'] = df['Date'].dt.strftime('%Y-%U')

#%% Semanal


df_gp = df.groupby('year_week').agg('last')
df_gp['pct_ret'] = df_gp['retorno acumulado'].pct_change() *100
# df_gp['pct_gsraii'] = df_gp['GSRAII Index'].pct_change() * 100

df_gp['pct_gsraii'] = (df_gp['GSRAII Index']-df_gp['GSRAII Index'].shift(1)) / np.abs(df_gp['GSRAII Index'].shift(1).values)


df_gp['pct_gsraii shift'] = df_gp['pct_gsraii'].shift(1).values

df_gp['GSRAII Index shift'] = df_gp['GSRAII Index'].shift(1).values

# ----CATEGORIAS-----
df_gp['category'] = np.where(df_gp['pct_ret'] <= -1.98, 1, 0)

df_gp['category'].value_counts()

desc = df_gp.describe()
desc_queda = df_gp[df_gp['category']==1].describe()
desc_maior = df_gp[df_gp['category']==0].describe()

desc_gsfiltrado = df_gp[df_gp['GSRAII Index']<=-1.12].describe()

df_gp.reset_index(inplace=True)

#%%


df_corr= df_gp[df_gp['category']==1].corr()

df_all = df_gp.merge(df, right_on='year_week', left_on='year_week' , how='outer')
[['category','year_week', 'Date_y', 'GSRAII Index_y', 'retorno_y']]
df_all = df_all[df_all['category']==0]

df_all_gp = df_all.groupby('year_week').agg(['std', 'sum', 'min', 'max'])


df_filtrado = df_all[(df_all['year_week']=='2020-12') | (df_all['year_week']=='2020-11')]





#%% Grafico

fig = px.line(df, x="Date", y=['GSRAII Index', 'retorno acumulado'])

# fig = px.line(df_gp, x="Date", y=['GSRAII Index', 'pct_ret'])

fig.add_trace(go.Scatter(
    x=df_gp[df_gp['category']==1]['Date'],
    y=df_gp[df_gp['category']==1]['pct_ret'],
    marker_size=10, mode='markers', name='Queda',
    ))

fig.show()


