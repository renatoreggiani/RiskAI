# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import quandl

#%%

"""# Captura e tratamento dos dados"""

def get_acwi():

    try:
        df = pd.read_csv('dados/ACWI.csv', parse_dates=['Date'])
        return df
    except FileNotFoundError:
        url_acwi = 'https://app2.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol=USD&index_variant=STRD&start_date=19970101&end_date=20211210&data_frequency=DAILY&baseValue=false&index_codes=892400'
        df = pd.read_excel(url_acwi, skiprows=6).dropna()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%b %d, %Y')
        df.rename(columns={df.columns[1]: df.columns[1].split()[0].lower()}, inplace=True)
        df.dropna(inplace=True)
        df.sort_values('Date', inplace=True)
        df.to_csv('dados\ACWI.csv', index=False)

        return df


def get_dxy():
    df = pd.read_csv('dados/DXY.csv', parse_dates=['Date'])
    df.dropna(inplace=True)
    df.sort_values('Date', inplace=True)
    df.rename(columns={df.columns[1]: df.columns[1].lower()}, inplace=True)
    
    return df
    

def get_fred(ticker, start='1996-01-01', end='2021-08-10'):
    if isinstance(ticker, list):
        df = pd.concat(
        [pdr.get_data_fred(serie,  start='1996-01-01', end='2021-08-10') for serie in ticker]
        , join='outer', axis=1)
    else:
        df = pdr.get_data_fred(ticker,  start='1996-01-01', end='2021-12-10')

    df.sort_index(inplace=True)
    df['year_week'] = df.index.strftime('%Y-%U')
    df = df.groupby('year_week').agg('last')

    return df


def get_sp500():
    df = pdr.get_data_fred('SP500',  start='1996-01-01', end='2021-12-10').dropna()
    df.sort_index(inplace=True)
    df['Date'] = df.index

    return df


def get_quandl(id_quandl, curve_diff=None):
    df = quandl.get(id_quandl, authtoken="Mw-vW_dxkPHHfjxjAQsF", start_date='1996-01-01')
    df.sort_index(inplace=True)
    
    if curve_diff:
        
        if len(curve_diff) == 2:
            col_name = f'{id_quandl}_{"-".join(curve_diff)}'
            df[col_name] = df[curve_diff[0]] - df[curve_diff[1]]
            df = df[[col_name]]
            df['year_week'] = df.index.strftime('%Y-%U')
            df = df.groupby('year_week').agg('last')
        else:
            raise 'Diferença deve ser calculada com 2 pontos'
    else:
        df.reset_index(inplace=True)
        df.rename(columns={'DATE':'Date'}, inplace=True)


    return df

