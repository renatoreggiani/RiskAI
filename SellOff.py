# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:04:02 2021

@author: F8564619
"""

import pandas as pd
import numpy as np

#%% Funcoes

def drawdown(return_series: pd.Series):
    """
    Retorna um DataFrame com
    valor, topo anterior, e percentual do drawdown
    """
    valor = (1+return_series).cumprod()
    topo_anterior = valor.cummax()
    drawdowns = (valor - topo_anterior)/topo_anterior
    return pd.DataFrame({"valor": valor-1,
                         "topo_anterior": topo_anterior-1,
                         "drawdown": drawdowns})




#%%
start = '2018-01-01'

url_acwi_etf = 'https://query1.finance.yahoo.com/v7/finance/download/ACWI?period1=1206662400&period2=1628380800&interval=1d&events=history&includeAdjustedClose=true'


url_acwi = 'https://app2.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol=USD&index_variant=STRD&start_date=19800101&end_date=20210810&data_frequency=DAILY&baseValue=false&index_codes=892400'

url_pmi = 'https://www.quandl.com/api/v3/datasets/ISM/MAN_PMI.csv?api_key=Mw-vW_dxkPHHfjxjAQsF'

url_trs = 'https://www.quandl.com/api/v3/datasets/USTREASURY/LONGTERMRATES.csv?api_key=Mw-vW_dxkPHHfjxjAQsF'
url_trs_curve = 'https://www.quandl.com/api/v3/datasets/USTREASURY/LONGTERMRATES.csv?api_key=Mw-vW_dxkPHHfjxjAQsF'

df_acwi_etf = pd.read_csv(url_acwi_etf, parse_dates=['Date']).sort_values('Date')
df_acwi_etf['ano_sem'] = df_acwi_etf['Date'].dt.strftime('%Y-%U')

df_acwi = pd.read_excel(url_acwi, skiprows=6)
df_acwi['Date'] = pd.to_datetime(df_acwi['Date'], errors='coerce', format='%b %d, %Y')
df_acwi.rename(columns={df_acwi.columns[1]: df_acwi.columns[1].split()[0].upper()}, inplace=True)

semanas = df_acwi['Date'].dt.strftime('%U-%Y').unique()


df_pmi = pd.read_csv(url_pmi, parse_dates=['Date']).sort_values('Date')


dd = drawdown(df_acwi['acwi'].pct_change().dropna())
dd.plot()


df_acwi.dtypes


#%%


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


qtd_amostras = 100

df = pd.DataFrame(
    {'real': np.random.randint(2, size=qtd_amostras),
     'previsto': np.random.randint(2, size=qtd_amostras),
     'probabilidade': np.random.random_sample(qtd_amostras)
     })

def validador(df):
    df = df.copy()

    confusion_matrix(df.iloc[:,0], df.iloc[:,1])
    f1_score(df.iloc[:,0], df.iloc[:,1])

def std_categorize(df, col='pct_change', n_std=1):
    std_range = df['std'].values[0] * n_std
    serie = (df[col] + 1).cumprod()
    if (serie.values[-1] >= (1 + 0.5*std_range)) & (np.max(serie) >= (1 + std_range)):
        return 1
    elif (serie.values[-1] <= (1 - 0.5*std_range)) & (np.min(serie) <= (1 - std_range)):
        return -1
    return 0

df_acwi.dropna(inplace=True)
df_gp = df_acwi.groupby('year_week').apply(std_categorize, ).reset_index(name='category')

