# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from permetrics.regression import Metrics
import seaborn as sns
from bayes_opt import BayesianOptimization
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adagrad,Adadelta,Adamax,Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout,BatchNormalization, LSTM ,GRU ,SimpleRNN,Bidirectional
import matplotlib.pyplot as plt
import time
tf.random.set_seed(516)
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
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
def split_hourly_data(df,t):
    train_mask = df["hour"] <= 16
    train=df[train_mask]
    train_X = train.drop(["AQI","hour"],axis =1)
    train_y = train['AQI']
    test_mask = df["hour"] >= 17
    test=df[test_mask]
    test_X = test.drop(["AQI","hour"],axis =1)
    test_y = test['AQI']
    train_y = train_y.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()
    return train_X,test_X,train_y,test_y
def do_pca(df):
    X_scaler = StandardScaler()
    train_X,test_X,train_y,test_y = split_hourly_data(df,t=0)
    train_y,test_y = pd.DataFrame(data = train_y),pd.DataFrame(data = test_y)
    train_X_scaled= X_scaler.fit_transform(train_X)
    test_X_scaled=X_scaler.transform(test_X)
    pca = PCA(n_components=0.9).fit(train_X_scaled)
    pca_train_x = pd.DataFrame(data = pca.transform(train_X_scaled))
    pca_test_x = pd.DataFrame(data = pca.transform(test_X_scaled))
    pca_x_concat = pd.concat([pca_train_x,pca_test_x], axis = 0)
    pca_y_concat = pd.concat([train_y,test_y], axis = 0)
    pca_x_concat.loc[:,10] = pca_y_concat
    pca_x_concat.loc[:,11] = df[['hour']]
    pca_x_concat.columns = ['1','2','3','4','5','6','7','8','9','AQI','hour']
    After_pca_df = pca_x_concat
    return After_pca_df
def make_DL_input(df,t_value):
    NoSTS = ['AQI','hour']        
    temp_df = df[[x for x in np.array(NoSTS,dtype=object) if x in df.columns.values]]
    df = df.drop(temp_df.columns.values,axis=1)
    colnt_df = df.columns.values
    for t in range(0,t_value-1):
        coln_temp = df.columns.values + "_t-{}".format(t+1)
        colnt_df = np.concatenate([coln_temp,colnt_df],axis=0)
    df = series_to_supervised(df,t_value-1, 1)
    df.columns = colnt_df
    df = pd.concat([df,temp_df[t_value-1:]],axis=1)
    train_X,test_X,train_y_ori,test_y_ori = split_hourly_data(df,t=t_value-1)
    train_y_ori, test_y_ori = train_y_ori.reshape(-1, 1), test_y_ori.reshape(-1, 1)
    X_scaler = MinMaxScaler(feature_range=(0,1)).fit(train_X)
    y_scaler = MinMaxScaler(feature_range=(0,1)).fit(train_y_ori)
    
    train_X = X_scaler.transform(train_X)
    test_X = X_scaler.transform(test_X)
    
    train_y = y_scaler.transform(train_y_ori)
    test_y = y_scaler.transform(test_y_ori)
    return train_X,test_X,train_y,test_y,train_y_ori,test_y_ori,X_scaler,y_scaler
#%%
def get_DL_result(df,name,init_points,n_iter,MYpatience,activationL,optimizerL,DL_params):
    cell = eval(name)
    best_params_dict = {}
    timedict = {}
    def model_for_bayes(neurons, activation, optimizer,
            learning_rate,  batch_size, epochs, hidden_layer,dropout_rate,time_step ): 
        optimizerD={'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
        'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
        'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
        'Nadam':Nadam(lr=learning_rate)}
        neurons = round(neurons)
        activation = activationL[round(activation)]
        optimizer = optimizerD[optimizerL[round(optimizer)]]
        batch_size = round(batch_size)
        epochs = round(epochs)
        time_step = round(time_step)
        hidden_layer = round(hidden_layer)
        if name == "Dense" :
            train_X,test_X,train_y,test_y,train_y_ori,test_y_ori,X_scaler,y_scaler = make_DL_input(df,1)

        else:
            # global fuck
            train_X,test_X,train_y,test_y,train_y_ori,test_y_ori,X_scaler,y_scaler = make_DL_input(df,time_step)
            train_X=train_X.reshape((train_X.shape[0], time_step, int(train_X.shape[1]/time_step)))
            # fuck = train_X 
        def DL_model():
            model = Sequential()
            h = 1  
            if name == "Dense":
                while h <= hidden_layer:
                    h = h +1
                    model.add(cell(units=neurons,
                            input_dim=(train_X.shape[1]),
                            # input_shape=(time_step, train_X.shape[2]),
                            activation=activation))
            else:
                while h < hidden_layer:
                    h = h +1
                    model.add(cell(units=neurons,
                            input_shape=(time_step, train_X.shape[2]),
                            activation=activation,
                            return_sequences=True))
                model.add(cell(units=neurons,
                            input_shape=(time_step, train_X.shape[2]),
                            activation=activation,
                            return_sequences=False))
            model.add(Dropout(dropout_rate))      
            model.add(Dense(1,activation='linear'))
            model.compile(loss='mean_absolute_error',metrics=['mae'],optimizer=optimizer)
            earlyStop=EarlyStopping(monitor="loss",verbose=0,mode='min',patience=MYpatience)
            model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,shuffle=False,
                      verbose = 2,callbacks=[earlyStop])
            return model
        model =DL_model()
        predictions = model.predict(train_X)
        global inv_yhat
        inv_yhat= y_scaler.inverse_transform(predictions.reshape(len(predictions),1))
        calperf = calPerformance(train_y_ori.flatten(),np.nan_to_num(inv_yhat).flatten()) 
        mape = -calperf[2]
        print("mape: %.5f" % (mape))
        return mape
    start = time.process_time()
    DL_bayesian = BayesianOptimization(model_for_bayes, DL_params, random_state=516)
    DL_bayesian.maximize(init_points=init_points, n_iter=n_iter)
    params_nn_ = DL_bayesian.max['params']
    learning_rate = params_nn_['learning_rate']
    params_nn_['activation'] = activationL[round(params_nn_['activation'])]
    params_nn_['batch_size'] = round(params_nn_['batch_size'])
    params_nn_['epochs'] = round(params_nn_['epochs'])
    params_nn_['hidden_layer'] = round(params_nn_['hidden_layer'])
    params_nn_['neurons'] = round(params_nn_['neurons'])
    params_nn_['dropout_rate'] = params_nn_['dropout_rate']
    params_nn_['time_step'] = round(params_nn_['time_step'])  
    optimizerD= {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                'Nadam':Nadam(lr=learning_rate)}
    params_nn_['optimizer'] = optimizerL[round(params_nn_['optimizer'])]
    params_nn_['optimizer'] = optimizerD[params_nn_['optimizer']]
    params_nn_
    if name == "Dense" :
        train_X,test_X,train_y,test_y,train_y_ori,test_y_ori,X_scaler,y_scaler = make_DL_input(df,1)
    else:
        train_X,test_X,train_y,test_y,train_y_ori,test_y_ori,X_scaler,y_scaler = make_DL_input(df,params_nn_['time_step'])
        train_X = train_X.reshape((train_X.shape[0], params_nn_['time_step'], int(train_X.shape[1]/params_nn_['time_step'])))
        test_X = test_X.reshape((test_X.shape[0], params_nn_['time_step'], int(test_X.shape[1]/params_nn_['time_step'])))
    def get_my_model():
        model = Sequential()
        h = 1   
        if name == "Dense":
            while h <= params_nn_['hidden_layer']:         
                h = h +1
                model.add(cell(units=params_nn_['neurons'],
                        input_dim=(train_X.shape[1]),
                        # input_shape=(params_nn_['time_step'], train_X.shape[2]),
                        activation=params_nn_['activation']))
        else:
            while h < params_nn_['hidden_layer']:         
                h = h +1
                model.add(cell(units=params_nn_['neurons'],
                        input_shape=(params_nn_['time_step'], train_X.shape[2]),
                        activation=params_nn_['activation'],                
                        return_sequences=True))
            model.add(cell(units=params_nn_['neurons'],
                        input_shape=(params_nn_['time_step'], train_X.shape[2]),
                        activation=params_nn_['activation'],
                        return_sequences=False))
        model.add(Dropout(params_nn_['dropout_rate']))
    
        model.add(Dense(1,activation='linear'))
        model.compile(loss='mean_absolute_error',metrics=['mae'],optimizer=params_nn_['optimizer'])
        earlyStop=EarlyStopping(monitor="loss",verbose=2,mode='min',patience=MYpatience)
        model.fit(train_X, train_y, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'],validation_data=(test_X, test_y),
                shuffle=False, verbose = 2,callbacks=[earlyStop])
        return model
    best_params_dict.setdefault(f"{name}: ",params_nn_)
    model = get_my_model()
    end = time.process_time()
    timedict.setdefault(f"{name}: ",(end-start))      
    

    train_pred =y_scaler.inverse_transform(model.predict(train_X).reshape(len(model.predict(train_X)), 1))
    test_pred =y_scaler.inverse_transform(model.predict(test_X).reshape(len(model.predict(test_X)), 1)) 
    train_perf = calPerformance(train_y_ori.flatten(), np.nan_to_num(train_pred).flatten())
    test_perf = calPerformance(test_y_ori.flatten(), np.nan_to_num(test_pred).flatten())

    performance = pd.DataFrame()
    performance["train_pred"] = pd.Series(train_pred.flatten())
    performance["test_pred"] = pd.Series(test_pred.flatten())
    performance[f"{name}_train_perf"] = pd.Series([round(x, 4) for x in train_perf ])
    performance[f"{name}_test_perf"]= pd.Series([round(x, 4) for x in test_perf ])
    print(best_params_dict)
    print(train_perf)
    print(test_perf)
    ## Save parameters and performances
    # model.save(f"Best_params_{name}.h5")
    # performance.to_csv(f"{name}_performance.csv", index=True)
    # file = open(f'{name}_bestParams.txt', 'w') 
    # for k,v in best_params_dict.items():
    #  	file.write(str(k)+' '+str(v)+'\n')
    # file.close()
    # file = open(f'{name}_time.txt', 'w') 
    # for k,v in timedict.items():
    #  	file.write(str(k)+' '+str(v)+'\n')
    # file.close()
#%%
activationL = ['relu','tanh','sigmoid']  #,'elu','selu','sigmoid','tanh' ,'sigmoid','tanh','sigmoid','tanh'
optimizerL = ['Adam','RMSprop','SGD']  #,'RMSprop','SGD','RMSprop','SGD'
DL_params ={
        'hidden_layer':(1,5),
        'neurons': (10,100),
        'activation':(0,len(activationL)-1),
        'optimizer':(0,len(optimizerL)-1),
        'learning_rate':(0.001, 0.01), 
        'time_step':(1,1),
        'batch_size':(200, 3000),
        'epochs':[500,500],
        'dropout_rate':(0.1,0.2)
        }
#%% do
ori_path = os.path.abspath(os.path.dirname(__file__)) #設定目前檔案位址為路徑
os.chdir(ori_path) #設定路徑
df = pd.read_csv('Hourly_Air_Data(Nanzi).csv')
df =df.drop(['MonitorDate'],axis =1).astype('float64')
### pca
# label_name = "pca"
# df = do_pca(df)
# df.to_csv("pca_data.csv", index=True)
### kpi
# label_name = "kpi"
df = df[['PM2.5','PM10','O3','AMB_TEMP','RH','NO','WD_HR','CO','AQI','hour']]
#%% do dl
for c in ['Dense']:  #'SimpleRNN','LSTM','GRU'
    # os.makedirs(label_name, exist_ok = True)  #創建label_name資料夾
    # os.chdir(label_name)    #設定label_name路徑
    # os.makedirs(c, exist_ok = True) #創建c資料夾
    # os.chdir(c) #設定c路徑
    get_DL_result(df,c,init_points=3,n_iter=2,MYpatience =50,activationL=activationL,optimizerL=optimizerL,DL_params=DL_params)







