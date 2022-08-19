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
library(e1071)
# cv_control = trainControl(
#   method = "cv", # cv = crovss validation
#   number = 3,  
#   allowParallel = TRUE
# )
set.seed(123)  # for reproducibility
data<-read.csv('空汙經濟指標(seasonal).csv',stringsAsFactors = T)
# data<-read.csv('空汙經濟指標(seasonal).csv',stringsAsFactors = T)
# data = data[1:61,]

df = data[,!(names(data) %in% c('Year','SF_factor'))]
x = data[,!(names(data) %in% c('Year','AQImean','SF_factor'))]
y = data$AQImean


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
MARS<-earth(y~., degree = 3,data=x,penalty = 0)  #,penalty = 0 ,nprune =12
evimp(MARS, trim=F)
cat(format(MARS))
summary(MARS)  ##ROE
evimp(MARS, trim=F)
MARS_pred<- predict(MARS,x)
(MARS_RMSE<-rmse(y,MARS_pred))
(MARS_MAE<-mae(y,MARS_pred))
(MARS_MAPE<-mape(y,MARS_pred))
write.csv(evimp(MARS, trim=F), file ="variable_NEWselection(seasonal)/AQI_MARS_impotance(R)_amount.csv", row.names = T)
## RF
rf_grid = expand.grid(
  mtry = 1:20%>% floor()
)

ctrl  <- trainControl(method  = "cv",number  = 10) #, summaryFunction = multiClassSummary

# random forest

fit.cv <- train(AQImean ~ ., data = df, method = "rf",
                trControl = ctrl, 
                tuneLength = 50)

print(fit.cv)
plot(fit.cv)
fit.cv$results

print(varImp(fit.cv)) # Variable importance
plot(varImp(fit.cv))
rf_pred<- predict(fit.cv,x)
(rf_RMSE<-rmse(y,rf_pred))
(rf_MAE<-mae(y,rf_pred))
(rf_MAPE<-mape(y,rf_pred))

cv_rf = train(
  AQImean ~., 
  data = df, 
  method = "rf",
  trControl = trainControl(method = "cv", number = 3),
  tuneGrid = rf_grid
)
mtr=c(6,8,10,12,15,20) #mtry: number of variables
ntr=c(1,3,5,7,9,11)#ntree: number of trees
nod=c(5,10,12) #nodesize: number of min samples
cv_rf$bestTune
tune.rf<-tune.randomForest(x=x, y=y, mtry=mtr, ntree=ntr, nodesize=nod)
tune.rf$best.model
tune.rf$best.parameters

# AQI.rf<-randomForest(AQImean~., data=df,mtry = cv_rf$bestTune$mtry, importance=T)
AQI.rf<-randomForest(AQImean~., data=df, importance=T,mtry = 10) #
print(varImp(AQI.rf))
AQI.rf$importanceSD
rf_pred<- predict(AQI.rf,x)
(rf_RMSE<-rmse(y,rf_pred))
(rf_MAE<-mae(y,rf_pred))
(rf_MAPE<-mape(y,rf_pred))
write.csv(AQI.rf$importanceSD, file ="variable_NEWselection(seasonal)/AQI_RF_impotance(R)_amount.csv", row.names = T)
####找最佳參數
####################
#######XGB##########
####################
# train_matrix <- sparse.model.matrix(AQImean ~ .-1, data = df)
xgb.train = xgb.DMatrix(data = as.matrix(df[,-12]),label = as.matrix(df$AQImean))
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
  AQImean ~., 
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
xgb.pred<-predict(xgb,as.matrix(df[,-12], reshape = T))
(xgb_RMSE<-rmse(y,xgb.pred))
(xgb_MAE<-mae(y,xgb.pred))
(xgb_MAPE<-mape(y,xgb.pred))
xgb_imp = xgb.importance(colnames(x), model = xgb)##查看變數重要性
xgb.plot.importance(xgb_imp)##視覺化
write.csv(xgb_imp, file ="variable_NEWselection(seasonal)/AQI_XGB_impotance(R)_amount.csv", row.names = T)
