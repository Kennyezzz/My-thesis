library(tidyverse)
library(zoo)
library(readr)
library(data.table)
library(tidyr)
library(reshape2)
data <- read.csv('HsCh_2019to2021.csv') #, fileEncoding="UTF-8"
data = data[!duplicated(data[,c(3,4)]),]%>%select(!c(SiteId,ItemId))
#寬轉長——melt
mydata1<-melt(
  data,
  id.vars=c("ItemEngName","MonitorDate"),#要保留的主欄位  
  variable.name = "hour",#0000000000000轉換後的分類欄位名稱（維度）
  value.name = "value" #轉換後的度量值名稱
)

final = mydata1  %>%spread(key =ItemEngName,value = value)
write.csv(final,"HsCh_2019to2021_melt.csv")



dataR <- read.csv('HsCh_2019to2021_melt.csv')
dataR <- final
# dataR <- dataC #read.csv('hourly_Data_CM2019.csv', fileEncoding = 'UTF-8')
# M <- dataR$MonitorDate
dataR$MonitorDate <- as.Date(dataR$MonitorDate)
dataR$Weekday <- as.factor(ifelse(weekdays(dataR$MonitorDate) %in% c("星期六","星期日"),print(0),print(1)))

m <- months(dataR$MonitorDate)
winter <- ifelse(m %in% c("十二月","一月","二月"), print(1),print(0))
spring <- ifelse(m %in% c("三月","四月","五月"), print(1),print(0))
summer <- ifelse(m %in% c("六月","七月","八月"), print(1),print(0))
fall <- ifelse(m %in% c("九月","十月","十一月"), print(1),print(0))

dataR$Winter <- as.factor(winter)
dataR$Spring <- as.factor(spring)
dataR$Summer <- as.factor(summer)
dataR$Fall <- as.factor(fall)
# dataR$MonitorDate = M
write.csv(dataR,"HsCh_hourly_Data_19to21_WdSe.csv", row.names = F) #, fileEncoding = 'UTF-8'


#check NAs
sort(sapply(dataR, function(x) sum(is.na(x))),dec = T)

pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(dataR,2,pMiss),dec=T)

