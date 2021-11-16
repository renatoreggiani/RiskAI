# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 22:13:19 2021

@author: F8564619
"""

import pandas as pd
import numpy as np

from captura import get_acwi, get_fred, get_pmi_us_classified, get_quandl

#%% Captura dos dados


def classe_gsrai(df, up_down=0, limit=0):

    df['gsrai_gt_up'] = np.where((df['GSRAII_diff'] > up_down) & (df['GSRAII Index'] >= limit), 1, 0)
    df['gsrai_gt_down'] = np.where((df['GSRAII_diff'] < up_down) & (df['GSRAII Index'] >= limit), 1, 0)
    df['gsrai_lt_up'] = np.where((df['GSRAII_diff'] > up_down) & (df['GSRAII Index'] < limit), 1, 0)
    df['gsrai_lt_down'] = np.where((df['GSRAII_diff'] < up_down) & (df['GSRAII Index'] < limit), 1, 0)

    regras = ['gsrai_gt_up', 'gsrai_gt_down', 'gsrai_lt_up', 'gsrai_lt_down']
    df['classe'] = np.nan
    for col in regras:
        df.loc[df[col]==1, 'classe'] = col
    return df.dropna()


def get_gsrai(up_down=0, diff_gsrai=0, limit=0, periods=1):

    df = pd.read_csv('dados/GSRAII.csv', parse_dates=['Date']).sort_values('Date')
    df['year_week'] = df['Date'].dt.strftime('%Y-%U')
    df = df.groupby('year_week').agg('last')
    df.drop(columns='Date', inplace=True)
    df['GSRAII_diff'] = df['GSRAII Index'].diff(periods)
    # df = classe_gsrai(df)
    future_diff = df['GSRAII_diff'].shift(-periods)
    df['category'] = np.where(future_diff < diff_gsrai, 1, 0)

    return df


def get_datas(return_df=False, scaler=False, diff_gsrai=0, periods=1):

    try:
        df = pd.read_csv('dados/Capturas.csv', index_col='year_week')

    except FileNotFoundError:
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
        df = pd.concat([df_fred, df_acwi, df_pmi, df_quandl],
                       join='outer', axis=1).sort_index()
        df.to_csv('dados/Capturas.csv')

    df_gsrai = get_gsrai(diff_gsrai=diff_gsrai, periods=periods)
    # Unifica capturas
    df = pd.concat([df, df_gsrai[['GSRAII Index', 'GSRAII_diff', 'category']] ],
                   join='outer', axis=1).sort_index()
    df = df.ffill().dropna()

    if scaler:
        scaler.fit(df)
        df_scale = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)
        df_scale['category'] = df['category']
        df = df_scale

    if return_df:

        return df

    else:
        X = df.drop(columns=['category'])
        y = df['category']

        return X, y


#%%
if __name__ == '__main__':

    from sklearn.preprocessing import MinMaxScaler

    df_model = get_datas(return_df=True, periods=3)


    X, y = get_datas(scaler=MinMaxScaler())


    scaler=MinMaxScaler()
    scaler.fit(df_model.drop(columns='category'))
    scaler.transform(df_model.drop(columns='category'))
