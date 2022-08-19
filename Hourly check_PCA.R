library(dplyr) 


data = read.csv('Hourly_Air_Data_Nanzi.csv') 
df = data %>% select(-MonitorDate) 
train = df %>% filter(hour<17)%>% select(-hour)
test = df %>% filter(hour>16) %>% select(-hour)


#標準化 
s_df = cbind(as.data.frame(scale(select(train,-AQI))),AQI=train$AQI) 

######### 
###PCA### 
######### 
#Rotation 
AQI.pca<-prcomp(~., data=select(s_df,-AQI),center=F, scale=F) 
AQI.pca 
summary(AQI.pca) 
eig1 = AQI.pca$sdev^2  #eigenvalues 
var1 = (eig1) / sum((eig1))   #計算變異量 
cum1 = cumsum((eig1) /sum((eig1)))  #計算累積變異量 
print(cbind(eig1,var1,cum1)) 
write.csv(cbind(eig1,var1,cum1), file="PCA_monthly.csv",row.names = F)
eigvec = AQI.pca$rotation #eigenvectors 
print(eigvec) 
write.csv(eigvec, file="PCA_monthly_eigvec.csv",row.names = T)
#將資料降維 已降到3維為例 
PCA_tr = data.frame(as.matrix(select(s_df,-AQI)) %*% as.matrix(eigvec[,1:6])) 
# PCA_tr$AQI = s_df$AQI 
head(PCA_tr)
