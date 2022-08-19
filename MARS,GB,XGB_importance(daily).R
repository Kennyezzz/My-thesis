library(gbm)
library(caret)
library(glmnet)
library(Metrics)
library(caret)
library(earth)
library(dplyr)
library(glue)
library(randomForest)
library(xgboost)
library("Matrix")
set.seed(123)  # for reproducibility

data<-read.csv('Daily_Air_Data_HsCh.csv',stringsAsFactors = T)
# df = data[,!(names(data) %in% c('MonitorDate'))]
df = data[,!(names(data) %in% c('MonitorDate'))]
x = data[,!(names(data) %in% c('AQI','MonitorDate'))]
y = data$AQI
## MARS
hyper_grid <- expand.grid(
  degree = 1:5%>% floor(),
  nprune = seq(1, 100, length.out = 10)%>% floor()
  # penalty = 0:5%>%floor()
)

cv_mars <- train(
  x = x,
  y = y,
  method = "earth",
  metric = "MAE",
  trControl = trainControl(method = "cv", number = 3),
  tuneGrid = hyper_grid
)
cv_mars$bestTune
MARS<-earth(y~., degree = cv_mars$bestTune$degree,trace=2,data=x,penalty = 0 )  #,penalty = 0 
cat(format(MARS))
summary(MARS)  ##ROE
evimp(MARS, trim=F)
MARS_pred<- predict(MARS,x)
(MARS_RMSE<-rmse(y,MARS_pred))
(MARS_MAE<-mae(y,MARS_pred))
(MARS_MAPE<-mape(y,MARS_pred))
write.csv(evimp(MARS, trim=F), file =glue("importance/AQI_MARS_importance(R)_t{t_value}.csv"), row.names = T)

## RF
rf_grid = expand.grid(
  .mtry=c(1:15)
)

cv_rf = train(
  AQI ~., 
  data = df, 
  method = "rf",
  trControl = trainControl(method = "cv", number = 3),
  tuneGrid = rf_grid
)
AQI.rf<-randomForest(AQI~., data=df,mtry = 5, importance=T)
                     # ntree=30, proximity=T, na.action=na.omit) cv_rf$bestTune$mtry
AQI.rf$importanceSD
rf_pred<- predict(AQI.rf,x)
(rf_RMSE<-rmse(y,rf_pred))
(rf_MAE<-mae(y,rf_pred))
(rf_MAPE<-mape(y,rf_pred))
write.csv(AQI.rf$importanceSD, file =glue("importance/AQI_RF_importance(R)_t{t_value}.csv"), row.names = T)
####т程ㄎ把计
####################
#######XGB##########
####################
# train_matrix <- sparse.model.matrix(AQImean ~ .-1, data = df)
xgb.train = xgb.DMatrix(data = as.matrix(df[,-16]),label = as.matrix(df$AQI))
###########xgb
xgbtree_grid = expand.grid(
  nrounds = 100, 
  max_depth = c(3,5,10), 
  eta = c(0.3, 0.03, 0.003), 
  gamma = c(0.01,0.1), 
  colsample_bytree = c(0.6, 0.7,0.8), 
  min_child_weight = 1,
  subsample = 1
)
xgbtree_trcontrol = trainControl(
  method = "cv", # cv = crovss validation
  number = 3,  
  allowParallel = TRUE,
  search="random"
)
cv_xgb = train(
  AQI ~., 
  data = df, 
  method = "xgbTree",
  trControl = xgbtree_trcontrol,
  tuneGrid = xgbtree_grid
)
summary(cv_xgb)
xgb <- xgboost(data = xgb.train, max_depth=cv_xgb$bestTune$max_depth, 
               eta=cv_xgb$bestTune$eta,gamma=cv_xgb$bestTune$gamma,
               colsample_bytree = cv_xgb$bestTune$colsample_bytree,
               min_child_weight = cv_xgb$bestTune$min_child_weight,
               objective='reg:linear', nround=cv_xgb$bestTune$nrounds)
xgb.pred<-predict(xgb,as.matrix(df[,-16], reshape = T))
(xgb_RMSE<-rmse(y,xgb.pred))
(xgb_MAE<-mae(y,xgb.pred))
(xgb_MAPE<-mape(y,xgb.pred))
xgb_imp = xgb.importance(colnames(x), model = xgb)##琩跑计璶┦
xgb.plot.importance(xgb_imp)##跌谋て
write.csv(xgb_imp, file =glue("importance/AQI_XGB_importance(R)_t{t_value}.csv"), row.names = T)

