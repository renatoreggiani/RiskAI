#%% Importação das bibliotecas

from preprocess import get_datas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#%% Chamada das variáveis

X, y = get_datas(scaler=MinMaxScaler())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

k_evaluation = pd.DataFrame(columns = ['k', 'score'])


#%% Criação do primeiro modelo: definindo melhor valor de k

for k in range(1, 50, 2):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    k_evaluation.loc[len(k_evaluation)] = [k, model.score(X_test, y_test)]
    
    