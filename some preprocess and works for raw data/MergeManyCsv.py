# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
#%%
Folder_Path = r'C:\Users\LAB_Kenny\OneDrive - National Chiao Tung University\LAB\Air_Data\original_files\AQI\2020'     #Nanzi_hourly\2021'          #要拼接的資料夾及其完整路徑，注意不要包含中文
SaveFile_Path =  r'C:\Users\LAB_Kenny\OneDrive - National Chiao Tung University\LAB\Air_Data\original_files\AQI\2020'  #Nanzi_hourly\2021'       #拼接後要儲存的檔案路徑
SaveFile_Name = r'Nanzi_2020all.csv'              #合併後要儲存的檔名

#修改當前工作目錄
os.chdir(Folder_Path)
#將該資料夾下的所有檔名存入一個列表
file_list = os.listdir()

#讀取第一個CSV檔案幷包含表頭
df = pd.read_csv(Folder_Path +'\\'+ file_list[0])   #編碼預設UTF-8，若亂碼自行更改, encoding= 'unicode_escape' , on_bad_lines='skip'
mask = df["SiteId"] == 53  #24 新竹 , 53	楠梓
df =df[mask]
#將讀取的第一個CSV檔案寫入合併後的檔案儲存
df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False)

#迴圈遍歷列表中各個CSV檔名，並追加到合併後的檔案
for i in range(1,len(file_list)):
    df = pd.read_csv(Folder_Path + '\\'+ file_list[i])
    mask = df["SiteId"] == 53
    df =df[mask]
    print(file_list[i])
    df.drop_duplicates()
    df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,index=False, header=False, mode='a+') #,encoding="utf_8_sig"

#%%
df = pd.read_csv('Hsch_AirX_2020to2021.csv',encoding="utf_8_sig") 
df.drop_duplicates()
df.to_csv('Hsch_AirX_2020to2021.csv',encoding="utf_8_sig")

#%% second way
import os, glob
import pandas as pd
#%%
for p in ['2021_12']:
    path = f"{p}/"
    
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged = df_merged[['MonitorDate','SiteId','AQI']] #,'AQI'
    # if p == '2020':
    #     df_merged = pd.concat(df_from_each_file, ignore_index=True)
    #     df_merged = df_merged[['MonitorDate','SiteId','AQI']] #,'AQI'
    #     # mask = df_merged["SiteId"] == 24  #24 新竹 , 53	楠梓
    #     # df_merged =df_merged[mask]
    # else:
    #     temp = pd.concat(df_from_each_file, ignore_index=True)
    #     temp = temp[['MonitorDate','SiteId','AQI']] #,'AQI'
    #     df_merged = pd.concat([df_merged,temp], ignore_index=True)
    #     # mask = df_merged["SiteId"] == 24  #24 新竹 , 53	楠梓
    #     # df_merged =df_merged[mask]
df_merged.to_csv( "202112_new.csv",encoding="utf_8_sig")