library(dplyr)
library(lubridate)

r_type = c('HsCh')

for (i in r_type)
{
  print(i)
  data <- read.csv(c(paste0(i,'_hourly_Data_19to21_WdSe.csv')))
  data$Wd = month(data$MonitorDate)
  all = data %>% select(PM2.5) %>% unlist() %>% as.numeric() %>% sum(na.rm=T)/4
  
  winter = data %>% filter(Wd%in%c(12,1,2)) %>% select(PM2.5) %>% unlist() %>% as.numeric() %>% sum(na.rm=T)/all 
  spring = data %>% filter(Wd%in%c(3,4,5)) %>% select(PM2.5) %>% unlist() %>% as.numeric() %>% sum(na.rm=T)/all 
  summer = data %>% filter(Wd%in%c(6,7,8)) %>% select(PM2.5) %>% unlist() %>% as.numeric() %>% sum(na.rm=T)/all 
  fall = data %>% filter(Wd%in%c(9,10,11)) %>% select(PM2.5) %>% unlist() %>% as.numeric() %>% sum(na.rm=T)/all 
  
  sf = c(winter,spring,summer,fall)
  print(sf)
  season = data %>% select(Winter:Fall)
  
  sf_table = apply(season,1,function(x){sf[as.logical(x)]})
  
  data_sf = cbind(data, Seasonal=sf_table) %>% select(-Wd,-Winter,-Spring,-Summer,-Fall)
  write.csv(data_sf,paste0(i,'_hourly_Data_19to21_WdSF.csv'), fileEncoding = 'UTF-8',row.names =F)
}


