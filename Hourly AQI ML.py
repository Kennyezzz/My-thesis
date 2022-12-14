# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pyearth import Earth
from permetrics.regression import Metrics
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
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
#%%
df = pd.read_csv('Hourly_Air_Data(Nanzi).csv')
df = df.drop(['MonitorDate'],axis =1).astype('float64')
kpi = ['PM2.5','PM10','O3','AMB_TEMP','RH','NO','WD_HR','CO']
train_mask = df["hour"] <= 16
train=df[train_mask]
train_X = train.drop(["AQI","hour"],axis =1)
train_y = train['AQI']
test_mask = df["hour"] >= 17
test=df[test_mask]
test_X = test.drop(["AQI","hour"],axis =1)
test_y = test['AQI']
## KPI
train_X = train_X[kpi]
test_X = test_X[kpi] 
# ## PCA
# train_X,test_X = do_pca(train_X,test_X)
train_y = train_y.to_numpy()
test_y = test_y.to_numpy()
#%% scale for SVR
X_scaler = preprocessing.StandardScaler()
train_X_scaled= X_scaler.fit_transform(train_X)
test_X_scaled=X_scaler.transform(test_X)
#%% MARS
param_MARS = {
            'max_degree': np.arange(1,3, 1),
            'penalty':np.arange(0,3,1),
             }
MARS_par = Earth(feature_importance_type='gcv') 
# MARS_rs = RandomizedSearchCV(MARS_par, param_MARS, cv=2,random_state=516,n_iter=5)
MARS_rs = GridSearchCV(MARS_par, param_MARS, cv=2)
MARS_rs.fit(train_X, train_y)
MARS_best = MARS_rs.best_estimator_
MARSpred_train = MARS_best.predict(train_X)
MARSpred_test = MARS_best.predict(test_X)
perf_MARS_train =  calPerformance(train_y,MARSpred_train)
perf_MARS_test = calPerformance(test_y, MARSpred_test)
MARS = pd.DataFrame()
MARS["train_pred"] = pd.Series(MARSpred_train)
MARS["test_pred"] = pd.Series(MARSpred_test)
MARS["train_perf"] = pd.Series([round(x, 4) for x in perf_MARS_train ])
MARS["test_perf"]= pd.Series([round(x, 4) for x in perf_MARS_test ])
print(MARS_rs.best_params_) 
print(perf_MARS_train,perf_MARS_test) #RMSE,MAE,MAPE
#%% RF
params_RF = {
        ## random search
        'n_estimators': np.arange(50,150, 10),
        'max_depth': np.arange(1,10, 1),
        'max_features': ['auto',0.9,0.7,0.5,0.3],
        'min_samples_split': np.arange(1,20, 1),
        'min_samples_leaf': np.arange(1,20, 1),
        }
RF_par = RandomForestRegressor(random_state=516)
RF_rs = RandomizedSearchCV(RF_par, params_RF, cv=2,random_state=516,n_iter = 10)
# RF_rs = GridSearchCV(RF_par, params_RF, cv=3)
RF_rs.fit(train_X, train_y)
RF_best = RF_rs.best_estimator_
RFpred_train = RF_best.predict(train_X)
RFpred_test = RF_best.predict(test_X)
perf_RF_train =  calPerformance(train_y,RFpred_train)
perf_RF_test = calPerformance(test_y,RFpred_test )
RF = pd.DataFrame()
RF["train_pred"] = pd.Series(RFpred_train)
RF["test_pred"] = pd.Series(RFpred_test)
RF["train_perf"] = pd.Series([round(x, 4) for x in perf_RF_train ])
RF["test_perf"]= pd.Series([round(x, 4) for x in perf_RF_test ])
print(RF_rs.best_params_)
print(perf_RF_train,perf_RF_test)
#%% XGB
params_XGB = {
        'n_estimators': np.arange(50,150, 10),
        'learning_rate':np.random.uniform(low=0.001,high=0.1,size=100).flatten(),
        'max_depth': np.arange(1,6,1),
        'min_child_weight':range(1,10,1), 
        'gamma':  [i/10.0 for i in range(0,9)], 
        'subsample': [i/10.0 for i in range(1,9)],  
        'colsample_bytree': [i/10.0 for i in range(1,9)],
        }
xgb_para = XGBRegressor(nthread=-1,eval_metric='mae',random_state=516)
XGB_rs = RandomizedSearchCV(xgb_para, params_XGB, cv=2, random_state=516,n_iter = 10)
# XGB_rs = GridSearchCV(xgb_para, params_XGB, cv=3)
XGB_rs.fit(train_X, train_y)
XGB_best = XGB_rs.best_estimator_
XGB_best.fit(train_X, train_y)
XGBpred_train = XGB_best.predict(train_X)
XGBpred_test = XGB_best.predict(test_X)
perf_XGB_train =  calPerformance(train_y,XGBpred_train)
perf_XGB_test = calPerformance(test_y,XGBpred_test)
XGB = pd.DataFrame()
XGB["train_pred"] = pd.Series(XGBpred_train)
XGB["test_pred"] = pd.Series(XGBpred_test)
XGB["train_perf"] = pd.Series([round(x, 4) for x in perf_XGB_train ])
XGB["test_perf"]= pd.Series([round(x, 4) for x in perf_XGB_test ])
print(XGB_rs.best_params_)
print(perf_XGB_train,perf_XGB_test)
#%% SVR
params_SVR = {
    'C': np.arange(1,5,1),
    'gamma': range(1,11,1),
    'kernel':['linear','rbf'],   
    'epsilon': np.arange(1,5,1)
    }
SVR_para = SVR() 
SVR_rs = RandomizedSearchCV(SVR_para, params_SVR, cv=2, random_state=516,n_iter =10)
# SVR_rs = GridSearchCV(SVR_para, params_SVR, cv=3)
SVR_rs.fit(train_X_scaled, train_y)
SVR_best = SVR_rs.best_estimator_  
SVR_best.fit(train_X_scaled, train_y)
SVRpred_train = SVR_best.predict(train_X_scaled)
SVRpred_test = SVR_best.predict(test_X_scaled)
perf_SVR_train =  calPerformance(train_y,SVRpred_train)
perf_SVR_test = calPerformance(test_y,SVRpred_test)
SVR = pd.DataFrame()
SVR["train_pred"] = pd.Series(SVRpred_train)
SVR["test_pred"] = pd.Series(SVRpred_test)
SVR["train_perf"] = pd.Series([round(x, 4) for x in perf_SVR_train ])
SVR["test_perf"]= pd.Series([round(x, 4) for x in perf_SVR_test ])
print(SVR_rs.best_params_)
print(perf_SVR_train,perf_SVR_test)
#%%
## ????????????????????????
plt.figure(figsize=(60,20))
plt.plot(test_y, marker='.', label="Test_actual",linewidth=5,markersize=20)
for i in ["MARS","RF","XGB","SVR"]:
    plt.plot(eval(i)["test_pred"],label=f"{i}_prediction",linewidth=5,markersize=20)
plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.ylabel('AQI', size=50)
plt.xlabel('Testing prediction', size=50)
plt.legend(fontsize=50)
plt.show()
#%% ????????????
train_perf = pd.DataFrame()
test_perf = pd.DataFrame()
for i in ["MARS","RF","XGB","SVR"]: 
    train_perf[i] = eval(i)["train_perf"][0:3]
    test_perf[i] = eval(i)["test_perf"][0:3]
train_perf.index = ["RMSE", "MAE", "MAPE"]
test_perf.index = ["RMSE", "MAE", "MAPE"]
#%%
print("Traing performance")
print(train_perf)
print("Testing performance")
print(test_perf)
#%% save
for i in ["MARS","RF","XGB","SVR"]: 
    ## save performance and prediction
    eval(i).to_csv(f"{i}_perf.csv", index=True)
    ## save best parameters
    path = f"{i}_parameters.txt"
    f = open(path, 'w')
    print(eval(f"{i}_rs").best_params_,file=f)
    f.close()