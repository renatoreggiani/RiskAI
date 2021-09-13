# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:13:19 2021

@author: F8564619
"""

import pandas as pd
import numpy as np

from captura import get_acwi, get_fred, get_pmi_us_classified, get_quandl

#%% Captura dos dados

def get_datas(return_df=False, scaler=False):
    series_fred = ['FEDFUNDS', 'CPALTT01USM657N', 'VIXCLS', 'DGS10', 'AAA10Y',
                   'BAMLH0A0HYM2EY', 'GEPUCURRENT',  'DGS3MO']
    df_fred = get_fred(series_fred)
    df_acwi = get_acwi()
    df_pmi = get_pmi_us_classified()
    df_quandl = pd.concat([
        get_quandl("ML/EMCBI"),
        get_quandl('USTREASURY/HQMYC', curve_diff=('10.0', '20.0')),
        get_quandl('USTREASURY/YIELD', curve_diff=('10 YR', '20 YR')),
        ], join='outer', axis=1)

    df_gsrai = pd.read_csv('dados/GSRAII.csv', parse_dates=['Date']).sort_values('Date')
    df_gsrai['year_week'] = df_gsrai['Date'].dt.strftime('%Y-%U')
    df_gsrai = df_gsrai.groupby('year_week').agg('last')
    df_gsrai.drop(columns='Date', inplace=True)
    df_gsrai['category'] = np.where(df_gsrai['GSRAII Index'] < -0.01, 1, 0)
    df_gsrai['category'] = df_gsrai['category'].shift(-1)

    df = pd.concat(
    [df_gsrai[['category', 'GSRAII Index']], df_fred, df_acwi, df_pmi, df_quandl ]
    , join='outer', axis=1).sort_index()

    df = df.ffill().dropna()

    if return_df:
        if scaler:
            scaler.fit(df)
            df_scale = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
            df_scale['category'] = df['category']
            df = df_scale
        return df

    else:
        X = df.drop(columns='category').values
        y = df['category'].values

        if scaler:
            scaler.fit(X)
            X = scaler.fit_transform(X)

        return X, y
        #%%
if __name__ == '__main__':

    from sklearn.preprocessing import MinMaxScaler

    df_model = get_datas(return_df=True, scaler=MinMaxScaler())


    X, y = get_datas(scaler=MinMaxScaler())


    scaler=MinMaxScaler()
    scaler.fit(df_model.drop(columns='category'))
    scaler.transform(df_model.drop(columns='category'))
