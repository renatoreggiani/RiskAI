# -*- coding: utf-8 -*-
"""logDiffTest.ipynb

Script de TESTE para transformação e seleção de variáveis

Apagar/refatorar assim que discutirmos os resultados em grupo.

"""



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from preprocess import get_datas



#%% Semanal

# df_gp =  classe_gsrai(df=df_gp, limit=0, media_movel=3)
diff_gsrai = -0.1


df = get_datas(return_df=True, diff_gsrai=diff_gsrai)

# 'GSRAII' como variável dependente
df['y'] = df['GSRAII_diff'].shift(-1)
df.dropna(inplace=True)

# Posteriormente, fazer a limpeza e seleção das variáveis brutas

df = df.drop(columns=['GSRAII_diff', 
                     'category', 
                     'acwi_log_diff',
                     'pmi_us_gt_50_up',
                     'pmi_us_gt_50_down',
                     'pmi_us_lt_50_up',
                     'pmi_us_lt_50_down'])

# Método stepwise foward não aceita os caracteres '/' e '-'
df = df.rename({'USTREASURY/YIELD_10 YR-20 YR': 'USTREASURY_YIELD_10_YR_20_YR',
                'USTREASURY/HQMYC_10.0-20.0': 'USTREASURY_HQMYC_10_20'}, 
                axis='columns')


#%%

def weekly_return(df):
  df_weekly_return = df.copy()

  # Loop through each stock (while ignoring time columns with index 0)
  for i in df.columns[1:]:
    
    # Loop through each row belonging to the stock
    for j in range(1, len(df)):

      # Calculate the percentage of change from the previous day
      #df_weekly_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
      df_weekly_return[i][j] = np.log(df[i][j]/df[i][j-1])
    
    # set the value of first row to zero since the previous value is not available
    df_weekly_return[i][0] = 0
  
  return df_weekly_return

#%%

# Get weekly returns 
df_retornos_semamais = weekly_return(df)

df_retornos_semamais['y']  = df_retornos_semamais['GSRAII'].shift(-1)

cm = df_retornos_semamais.corr()

plt.figure(figsize=(10, 10))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax);

#%% Seleção de variáveis 

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model

#%% Aplicação regressão stepwise foward

# Após a transformação (log do quociente) das variáveis, houve incidência de 
# valores nulos (NA) e infinitos (-inf) - ANALISAR**

df_retornos_semamais.replace([np.inf, -np.inf], np.nan, inplace=True)
df_retornos_semamais.dropna(inplace=True)

model = forward_selected(df_retornos_semamais, 'y')

print (model.rsquared_adj)