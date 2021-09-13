# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import quandl

#%%

"""# Captura e tratamento dos dados"""

def get_acwi():

    try:
        df = pd.read_csv('dados/ACWI.csv', index_col='year_week')
        return df
    except FileNotFoundError:
        url_acwi = 'https://app2.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol=USD&index_variant=STRD&start_date=19970101&end_date=20210810&data_frequency=DAILY&baseValue=false&index_codes=892400'
        df = pd.read_excel(url_acwi, skiprows=6).dropna()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%b %d, %Y')

        df.rename(columns={df.columns[1]: df.columns[1].split()[0].lower()}, inplace=True)
        df['year_week'] = df['Date'].dt.strftime('%Y-%U')
        df.dropna(inplace=True)
        df.sort_values('Date', inplace=True)
        df = df.groupby('year_week').agg('last').reset_index()

        # df['acwi_pct_change'] = df['acwi'].pct_change()
        df['acwi_log_diff'] = np.log(df['acwi']/df['acwi'].shift(1))
        df = df.drop(columns=['Date', 'acwi']).set_index('year_week')
        df.to_csv('dados\ACWI.csv')

        return df


def get_pmi_us_classified():
    df = pd.read_csv('https://www.quandl.com/api/v3/datasets/ISM/MAN_PMI.csv?api_key=Mw-vW_dxkPHHfjxjAQsF',
                            parse_dates=['Date']).sort_values('Date')
    df['year_week'] = df['Date'].dt.strftime('%Y-%U')
    mean_3m = df['PMI'].rolling(3).mean()

    df['pmi_us_gt_50_up'] = np.where((df['PMI'] > mean_3m) & (df['PMI'] >=50), 1, 0)
    df['pmi_us_gt_50_down'] = np.where((df['PMI'] < mean_3m) & (df['PMI'] >=50), 1, 0)
    df['pmi_us_lt_50_up'] = np.where((df['PMI'] > mean_3m) & (df['PMI'] < 50), 1, 0)
    df['pmi_us_lt_50_down'] = np.where((df['PMI'] < mean_3m) & (df['PMI'] < 50), 1, 0)
    df = df.drop(columns=['Date', 'PMI']).set_index('year_week')
    return df


def get_fred(ticker):
    if isinstance(ticker, list):
        df = pd.concat(
        [pdr.get_data_fred(serie,  start='1996-01-01', end='2021-08-10') for serie in ticker]
        , join='outer', axis=1).ffill()
    else:
        df = pdr.get_data_fred(ticker,  start='1996-01-01', end='2021-08-10')

    df.sort_index(inplace=True)
    df['year_week'] = df.index.strftime('%Y-%U')
    df = df.groupby('year_week').agg('last')

    return df


def get_quandl(id_quandl, curve_diff=None):
    df = quandl.get(id_quandl, authtoken="Mw-vW_dxkPHHfjxjAQsF", start_date='1996-01-01').sort_index()

    if curve_diff:
        if len(curve_diff) == 2:
            col_name = f'{id_quandl}: {"-".join(curve_diff)}'
            df[col_name] = df[curve_diff[0]] - df[curve_diff[1]]
            df = df[[col_name]]
        else:
            raise 'Diferença deve ser calculada com 2 pontos'

    df['year_week'] = df.index.strftime('%Y-%U')
    df = df.groupby('year_week').agg('last')#.reset_index()

    return df


#%%

