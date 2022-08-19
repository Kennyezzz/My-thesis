# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import pandas as pd
import numpy as np
#from __future__ import division}
from sklearn.model_selection import train_test_split

#%%
#data = pd.read_csv('../hourly_Data_10to20_WdSe.csv')#,index_col="MonitorDate")
# data = pd.read_csv('../hourly_Data_2021top_WdSe.csv')
region='HsCh'
data = pd.read_csv(f"{region}_hourly_Data_19to21_WdSF.csv")
df = pd.DataFrame(data)
df['MonitorDate'] = pd.to_datetime(df['MonitorDate'])
date = df['MonitorDate']
df = df.set_index(['MonitorDate'],drop=True)
print('data shape: ', df.shape)

# First few rows of the training dataset
df.head(10)
df.info()
d=df.describe()

#%%
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

#%%
data_missing= missing_values_table(df)
data_missing

#%%
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from sklearn.impute import KNNImputer 
 
rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))
#%%
def optimize_k(data, target): 
    errors = [] 
    for k in range(1, 19, 1): 
        imputer = KNNImputer(n_neighbors=k) 
        imputed = imputer.fit_transform(df) 
        df_imputed = pd.DataFrame(imputed, columns=df.columns) 
         
        X = df_imputed.drop(target, axis=1) 
        y = df_imputed[target] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
        model = RandomForestRegressor() 
        model.fit(X_train, y_train) 
        preds = model.predict(X_test) 
        error = rmse(y_test, preds) 
        errors.append({'K': k, 'RMSE': error}) 
         
    return errors 
#%%
k_errors = optimize_k(df, target='PM2.5')
k_errors
#%%

imputer = KNNImputer(n_neighbors=3) 
imputed = imputer.fit_transform(df) 
df_imputed = pd.DataFrame(imputed, columns=df.columns)


data_missing= missing_values_table(df_imputed)
data_missing
df_imputed.head(10)


#%%
df_0=pd.concat([date,df_imputed],axis=1)
# df_0=df_0.drop(['Unnamed: 0'],axis=1)
df_0

#%%
print(df_0.dtypes)
df_0.isnull().sum()

#%%
#df_0.to_csv('CM_data_2011to2020_afterKNN.csv', index=False)
df_0.to_csv(f'{region}_hourly_Data_19to21_WdSF_afterKNN.csv', index=False)

