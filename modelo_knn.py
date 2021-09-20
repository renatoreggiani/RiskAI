#%% Anotações

# Tentar manter algumas das variáveis autocorrelacionadas 
# n_splits = 16, k = 3


#%% Importação das bibliotecas

from preprocess import get_datas

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np


#%% Extração de variáveis para testes

df = get_datas(return_df=True)

df_X = df.drop(columns = 'category')


#%% Criando de modelo dummy

X = MinMaxScaler().fit_transform(df_X)

y = df['category']

seed = 101

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = seed)

knn_m0 = KNeighborsClassifier(n_neighbors = 3)

knn_m0.fit(X_train, y_train)

print(knn_m0.score(X_test, y_test))


#%% Seleção de variáveis 1: Verificando multicolinearidade entre variáveis explicativas

def drop_var_multicol(df_var_X, lim = 0.09):
    
    """ Recebe as variáveis preditoras e encontra a multicolinearidade entre elas
    a partir de cálculos de autovalores e autovetores, eliminando as que apresentam 
    os maiores valores.
    Limite máximo de valor de autovetores default 0.09"""
    
    corr = np.corrcoef(df_var_X, rowvar = 0)
    
    eingvalues, eingvectors = np.linalg.eig(corr)
    
    min_eval = np.argmin(eingvalues)
    
    max_evecs = np.argwhere(eingvectors[:,min_eval] > lim)[:,0]
    
    drop_cols = df_var_X.columns[max_evecs]
                             
    df_var_X_final = df_var_X.drop(columns = drop_cols)
    
    return df_var_X_final

df_X_final = drop_var_multicol(df_X, lim = i)

X_s1 = MinMaxScaler().fit_transform(df_X_final)

X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(X_s1, 
                                                                y, 
                                                                test_size = 0.25, 
                                                                random_state = seed)


knn_m1 = KNeighborsClassifier(n_neighbors = 3)

knn_m1.fit(X_train_s1, y_train_s1)

print(knn_m1.score(X_test_s1, y_test_s1))
    

#%% Seleção de variáveis 2: Utilizando SelectKBest

def select_var_kbest(df_var_X, df_var_y):  

    k_eval = pd.DataFrame(columns = ['k', 'score', 'var_list'])
    
    def my_score(X, y):
        
        return mutual_info_classif(df_var_X, df_var_y, random_state = seed)
    
    for k in range(3,17,1):
        
        sel = SelectKBest(k=k, score_func = my_score
                          ).fit(df_var_X, df_var_y)
        
        var_index = sel.get_support(indices = True)
        
        var = list(df_X.iloc[:,var_index].columns)
        
        X_s2 = df_var_X[var]
        
        X_s2 = MinMaxScaler().fit_transform(X_s2)
        
        X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(X_s2, 
                                                                        df_var_y, 
                                                                        test_size = 0.25, 
                                                                        random_state = seed)
        
        knn_m2 = KNeighborsClassifier(n_neighbors = 3)
        
        knn_m2.fit(X_train_s2, y_train_s2)
        
        k_eval.loc[len(k_eval)] = [k, knn_m2.score(X_test_s2, y_test_s2), var]
        
    var = k_eval.sort_values('score', ascending = False
                             ).reset_index()['var_list'][0]
    
    df_var_X_final = df_var_X[var]
    
    return df_var_X_final

X_s2 = select_var_kbest(df_X, df['category'])
    
X_s2 = MinMaxScaler().fit_transform(X_s2)

X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(X_s2, 
                                                                y, 
                                                                test_size = 0.25, 
                                                                random_state = seed)
    
knn_m2 = KNeighborsClassifier(n_neighbors = 3)
    
knn_m2.fit(X_train_s2, y_train_s2)
    
print(knn_m2.score(X_test_s2, y_test_s2))  


#%% Seleção de variáveis 3: Utilizando RandomForestClassifier

def select_var_randomforest(df_var_X, df_var_y):  

    y_cod = LabelEncoder().fit_transform(df_var_y)
    
    sel = SelectFromModel(RandomForestClassifier(n_estimators = 1000,
                                                  random_state = seed)
                          ).fit(df_var_X, y_cod)
                                                                             
                                                                            
    var_index = sel.get_support(indices = True)
        
    var = list(df_X.iloc[:,var_index].columns)
    
    df_var_X_final = df_var_X[var]
    
    return df_var_X_final
    
X_s3 = select_var_randomforest(df_X, df['category'])
    
X_s3 = MinMaxScaler().fit_transform(X_s3)
    
X_train_s3, X_test_s3, y_train_s3, y_test_s3 = train_test_split(X_s3, 
                                                                y, 
                                                                test_size = 0.25, 
                                                                random_state = seed)
    
knn_m3 = KNeighborsClassifier(n_neighbors = 3)
    
knn_m3.fit(X_train_s3, y_train_s3)
    
knn_m3_score = knn_m3.score(X_test_s3, y_test_s3)
    
print(knn_m3_score)

                                                                         
#%% Testando ShuffleSplit

knn_m4 = KNeighborsClassifier(n_neighbors = 3)

n_eval = pd.DataFrame(columns = ['n', 'mean', 'std', 'min', 'max'])

for n in range(1, 1001, 1):
    
    ss = ShuffleSplit(n_splits=n, test_size=0.25, random_state = seed)
    
    scores = cross_val_score(knn_m4, X_s1, y, scoring='accuracy', 
                             cv=ss, n_jobs=-1)

    n_eval.loc[len(n_eval)] = [n, 
                               scores.mean(),
                               scores.std(),
                               scores.min(),
                               scores.max()]


n_eval['mean'].plot.line()

n_eval['std'].plot.line()

n_eval['min'].plot.line()

n_eval['max'].plot.line()


#%% Definindo melhor valor de k

k_eval = pd.DataFrame(columns = ['k', 'mean', 'std', 'min', 'max'])

for k in range(1, 50, 2):
    
    knn_m5 = KNeighborsClassifier(n_neighbors = k)
    
    ss = ShuffleSplit(n_splits=16, test_size=0.25, random_state = seed)
    
    scores = cross_val_score(knn_m5, X_s1, y, scoring='accuracy', 
                             cv=ss, n_jobs=-1)
    
    k_eval.loc[len(k_eval)] = [k, 
                               scores.mean(),
                               scores.std(),
                               scores.min(),
                               scores.max()]





