# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from permetrics.regression import Metrics
def do_pca(train_X,test_X):
    X_scaler = preprocessing.StandardScaler()
    train_X_scaled= X_scaler.fit_transform(train_X)
    test_X_scaled=X_scaler.transform(test_X)
    pca = PCA().fit(train_X_scaled)
    plt.rcParams["figure.figsize"] = (12,6)
    fig, ax = plt.subplots()
    xi = np.arange(1, train_X_scaled.shape[1]+1, step=1)
    y_for_pca = np.cumsum(pca.explained_variance_ratio_)
    def find_best_Number_of_Components():
        for i in xi:
            if y_for_pca[i-1] > 0.9:
                best_n = i
                return best_n
    plt.ylim(0.0,1.1)
    plt.plot(xi, y_for_pca[:], marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, train_X_scaled.shape[1]+1, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.text(0.5, 1.05, '90% cut-off threshold', color = 'red', fontsize=16)
    ax.grid(axis='x')
    plt.savefig('PCA_plot.png')
    plt.show()  
    pca = PCA(n_components=0.9).fit(train_X_scaled)
    train_X = pca.transform(train_X_scaled)
    test_X = pca.transform(test_X_scaled)
    return train_X,test_X
def calPerformance(y_true,y_pred):
    temp_list=[] 
    calPerf = Metrics(y_true, y_pred)
    RMSE_score = calPerf.root_mean_squared_error(clean=True, decimal=6)
    temp_list.append(RMSE_score)
    MAE_score = calPerf.mean_absolute_error(clean=True, decimal=6)
    temp_list.append(MAE_score)
    MAPE_score = calPerf.mean_absolute_percentage_error(clean=True, decimal=6)
    temp_list.append(MAPE_score)
    return temp_list
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM,GRU
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import tensorflow as tf
#%%
df = pd.read_csv('Daily_Air_Data(HsCh).csv')
df = df.set_index(['MonitorDate'],drop=True)
df = df.astype('float64')
train_start = '2020/1/1' 
train_end = '2021/6/30' 
test_start = '2021/7/1' 
test_end = '2021/12/31' 
train_y = df["AQI"].loc[train_start:train_end].to_numpy()
test_y = df["AQI"].loc[test_start:test_end].to_numpy()
## kpi"
kpi = ["PM2.5","O3","WS_HR","AMB_TEMP","PM10","CO","NO","NOx","NO2"]   
train_X = df[kpi].loc[train_start:train_end]
test_X = df[kpi].loc[test_start:test_end]
## pca
# train_X = df.drop("AQI",axis=1).loc[train_start:train_end]
# test_X = df.drop("AQI",axis=1).loc[test_start:test_end]
# train_X,test_X = do_pca(train_X,test_X)
#%% 
train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)
X_scaler = preprocessing.MinMaxScaler().fit(train_X)
y_scaler = preprocessing.MinMaxScaler().fit(train_y)

#%%
train_X= X_scaler.transform(train_X).reshape((train_X.shape[0], 1, int(train_X.shape[1]/1)))
test_X =X_scaler.transform(test_X).reshape((test_X.shape[0], 1, int(test_X.shape[1]/1)))
train_y_ori = train_y
test_y_ori = test_y
train_y = y_scaler.transform(train_y)
test_y = y_scaler.transform(test_y)
#%% Bulid NN model 
es = EarlyStopping(monitor='loss', mode='min', patience = 50, verbose=1)
def build_model(name="Dense",n_dropout=0.1,n_hidden=1,n_neurons=4,learning_rate=1e-2,
                activation='relu',kernel_initializer='uniform',n_epochs=10,
                n_batch_size=10,select_optimizer=optimizers.Adam):#
    
    model = Sequential()
    if name == "Dense":
        for layer in range(n_hidden):
            model.add(Dense(n_neurons,activation=activation,input_dim=train_X.shape[2],kernel_initializer=kernel_initializer))
    else:
        cell = eval(name)
        for layer in range(n_hidden-1):
            model.add(cell(units=n_neurons,input_shape=(train_X.shape[1], train_X.shape[2]),activation=activation,
                            kernel_initializer=kernel_initializer,return_sequences=True))
        model.add(cell(units=n_neurons,input_shape=(train_X.shape[1], train_X.shape[2]),activation=activation,
                kernel_initializer=kernel_initializer,return_sequences=False))
    model.add(Dropout(n_dropout))
    model.add(Dense(1, activation='linear'))
              
    optimizer = select_optimizer(lr=learning_rate)
    model.compile(loss='mae', metrics=['mae'],optimizer=optimizer)
    
    model.fit(train_X,train_y, epochs=n_epochs, batch_size=n_batch_size, callbacks=[es]) 
    return model
keras=KerasRegressor(build_model)
#%% set parameters
tf.random.set_seed(516)
np.random.seed(516)
# for what_NN in ['Dense','SimpleRNN','LSTM','GRU']: ## Make a for loop to auto run.
what_NN = 'SimpleRNN' # Choose what kind of NN :'Dense','SimpleRNN','LSTM','GRU'  
params = {
        "name": [what_NN],
        "n_dropout":Real(low=0.1, high=0.2, prior='log-uniform'),
        "n_hidden":Integer(low=1,high=5),
        "n_neurons":Integer(low=1,high=30),
        "kernel_initializer":Categorical(categories= ['uniform','normal']),
        "learning_rate":Real(low=0.001, high=0.01, prior='log-uniform'), 
        "activation":Categorical(categories= ['relu','elu','selu','sigmoid','tanh']),  
        "n_batch_size":Integer(low=10,high=500),
        "n_epochs":[500],
        "select_optimizer":[optimizers.Adam,optimizers.SGD,optimizers.RMSprop]  
        }
bayes_search_cv=BayesSearchCV(keras,params,n_iter=2,cv=2,random_state=516) 
bayes_search_cv.fit(train_X,train_y)
globals()[what_NN+'_best_model'] = bayes_search_cv.best_estimator_ ## This will be best model of BayesSearchCV
globals()[what_NN+'_best_parameters']= bayes_search_cv.best_params_
print(eval(what_NN+'_best_parameters'))
## Prediction
globals()[what_NN+'_test_pred'] = y_scaler.inverse_transform(eval(what_NN+'_best_model').predict(test_X).reshape(len(eval(what_NN+'_best_model').predict(test_X)), 1))
globals()[what_NN+'_train_pred'] = y_scaler.inverse_transform(eval(what_NN+'_best_model').predict(train_X).reshape(len(eval(what_NN+'_best_model').predict(train_X)), 1))
globals()[what_NN+'_train_perf'] = calPerformance(train_y_ori.flatten(), globals()[what_NN+'_train_pred'].flatten())
globals()[what_NN+'_test_perf'] = calPerformance(test_y_ori.flatten(), globals()[what_NN+'_test_pred'].flatten())
print(eval(what_NN+'_train_perf'),eval(what_NN+'_test_perf'))
###
#%% make performance and prediction list
# for what_NN in ['Dense','SimpleRNN','LSTM','GRU']:  ## Make a for loop to show all performances and predictions.
globals()[what_NN+'_pf']= pd.DataFrame()
globals()[what_NN+'_pf']["train_pred"] = pd.Series(eval(what_NN+"_train_pred").flatten())
globals()[what_NN+'_pf']["test_pred"] = pd.Series(eval(what_NN+"_test_pred").flatten())
globals()[what_NN+'_pf']["train_perf"] = pd.Series([round(x, 4) for x in eval(what_NN+"_train_perf") ])
globals()[what_NN+'_pf']["test_perf"]= pd.Series([round(x, 4) for x in eval(what_NN+"_test_perf") ])
###
## 畫實際值和預測值
plt.figure(figsize=(60,20))
plt.plot(test_y_ori, marker='.', label="Test_actual",linewidth=5,markersize=20)
# for what_NN in ['Dense','SimpleRNN','LSTM','GRU']: ## Make a for loop to plot all models.
plt.plot(eval(what_NN+'_test_pred'),label=f"{what_NN}_prediction",linewidth=5,markersize=20)
###
plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.ylabel('AQI', size=50)
plt.xlabel('Testing prediction', size=50)
plt.legend(fontsize=50)
plt.show()
#%% 績效總表
train_perf = pd.DataFrame()
test_perf = pd.DataFrame()
# for what_NN in ['Dense','SimpleRNN','LSTM','GRU']:  ## Make a for loop to show all performance.
train_perf[what_NN] = eval(what_NN+'_pf')["train_perf"][0:3]
test_perf[what_NN] = eval(what_NN+'_pf')["test_perf"][0:3]
###
train_perf.index = ["RMSE", "MAE", "MAPE"]
test_perf.index = ["RMSE", "MAE", "MAPE"]
print("Traing performance")
print(train_perf)
print("Testing performance")
print(test_perf)
#%% save
# for what_NN in ['Dense','SimpleRNN','LSTM','GRU']:  ## Make a for loop to save all....
## save performance and prediction
eval(what_NN+'_pf').to_csv(f"{what_NN}_pf.csv", index=True)
## save best parameters
path = f"{what_NN}_parameters.txt"
f = open(path, 'w')
print(eval(what_NN+'_best_parameters'),file=f)
f.close()