# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

#%%
df = pd.read_csv('Monthly_Air_Data_Taiwan.csv')
df_train = df.loc[0:48]
df_test = df.loc[49:63]

train_y = df_train[["AQImean"]].to_numpy().flatten()
test_y = df_test[["AQImean"]].to_numpy().flatten()

train_X = df_train.drop(['Year','AQImean'],axis=1)
test_X = df_test.drop(['Year','AQImean'],axis=1)
colname = train_X.columns.values
#%%
X_scaler = preprocessing.StandardScaler()
train_X_scaled= X_scaler.fit_transform(train_X)
test_X_scaled=X_scaler.transform(test_X)
#%%
# pca = PCA(n_components=0.9).fit(train_X_scaled)
pca = PCA().fit(train_X_scaled)
pca_train_X = pca.transform(train_X_scaled)
pca_test_X = pca.transform(test_X_scaled)
pca.explained_variance_
#%%
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = train_X.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df






