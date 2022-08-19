library(tidyverse)
library(lubridate)
library(gridExtra)




#data = read.csv("../hourly_Data_10to21_WdSe_afterLinear.csv", header = T)
#data = read.csv("../CD_hourly_Data_10to21_WdSe_afterlinear.csv", header = T)
data = read.csv("all.csv", header = T)

data.date <- as.Date(data$MonitorDate)
df = data.frame(year = year(data.date), 
                month = month((data.date)),
                yday = yday(data.date),
                mday = mday(data.date),
                wday = wday(data.date,label=T),
                #hour = rep(seq(0,23),nrow(data)/24),
                hour = data$hour,
                PM2.5 = data$PM2.5)
data = read.csv("merged_new.csv", header = T)
data.date <- as.Date(data$MonitorDate)
df = data.frame(year = year(data.date), 
                month = month((data.date)),
                #yday = yday(data.date),
                #mday = mday(data.date),
                #wday = wday(data.date,label=T),
                #hour = rep(seq(0,23),nrow(data)/24),
                AQI = data$AQI,
                SiteId = data$SiteId) 
monthly_2016aqi = df %>% filter(SiteId == 16|24|32|44|53)%>%filter(year == 2016)   %>% group_by(month)%>%  
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2016)
monthly_2017aqi = df %>%filter(year == 2017) %>% filter(SiteId == 16 |24|32|44|53)  %>% group_by(month)%>% 
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2017)
monthly_2018aqi = df %>%filter(year == 2018) %>% filter(SiteId == 16 |24|32|44|53)  %>% group_by(month)%>% 
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2018)
monthly_2019aqi = df %>%filter(year == 2019) %>% filter(SiteId == 16 |24|32|44|53)  %>% group_by(month)%>%  
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2019)
monthly_2020aqi = df %>%filter(year == 2020) %>% filter(SiteId == 16 |24|32|44|53)  %>% group_by(month)%>%  
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2020)
monthly_2021aqi = df %>%filter(year == 2021) %>% filter(SiteId == 16 |24|32|44|53)  %>% group_by(month)%>% 
  summarise(mean = mean(AQI),med = median(AQI))%>%mutate(year=2021)
z <- bind_rows(monthly_2016aqi, monthly_2017aqi,monthly_2018aqi,monthly_2019aqi,monthly_2020aqi,monthly_2021aqi)
write.csv(z,"selectedAQI.csv")
#歷年各天(365天)
d_yday = df %>% group_by(yday) %>% summarise(mean = mean(PM2.5),med = median(PM2.5))
ggplot(d_yday) + geom_line(aes(yday, mean),lwd=1,col=1) + geom_line(aes(yday, med),lwd=1,col=4) + 
  labs(title='歷年各天(365天)') + theme(plot.title = element_text(color=1, size=16, face="bold"))

# 各月(1~12) 
d_ym = df %>% group_by(month) %>% summarise(mean = mean(PM2.5),med = median(PM2.5))
d_ym$month <- as.factor(d_ym$month)
plot1 <- ggplot(d_ym) + geom_bar(aes(month, mean),stat='identity') + ylim(0,35) + labs(title='各月(1~12)平均')
plot2 <- ggplot(d_ym) + geom_bar(aes(month, med),stat='identity') + ylim(0,35) + labs(title='各月(1~12)中位數')
grid.arrange(plot1, plot2, nrow=2)

# 各週(週一、週二...)
d_wday = df %>% group_by(wday) %>% summarise(mean = mean(PM2.5),med = median(PM2.5))
d_wday$wday <- as.factor(d_wday$wday)
plot3 <- ggplot(d_wday) + geom_bar(aes(wday, mean),stat='identity') + ylim(0,25) + labs(title='各週(週一、週二...)平均')
plot4 <- ggplot(d_wday) + geom_bar(aes(wday, med),stat='identity') + ylim(0,25) + labs(title='各週(週一、週二...)中位數')
grid.arrange(plot3, plot4, nrow=2)

# 各小時(00~23)
d_hour = df %>% group_by(hour) %>% summarise(mean = mean(PM2.5),med = median(PM2.5))
d_hour$hour <- as.factor(d_hour$hour)
plot5 <- ggplot(d_hour) + geom_bar(aes(hour, mean),stat='identity') + ylim(0,25) + labs(title='各小時(00~23)平均')
plot6 <- ggplot(d_hour) + geom_bar(aes(hour, med),stat='identity') + ylim(0,25) + labs(title='各小時(00~23)中位數')
grid.arrange(plot5, plot6, nrow=2)









##OUTLIER
out <- boxplot.stats(df$PM2.5)$out
out_ind <- which(df$PM2.5 %in% c(out))
df_out <- df[out_ind,]

df_woout <- data[-out_ind,]
#write.csv(df_woout,"../hourly_Data_10to21_WdSe_afterLinear_woOutlier.csv")
#write.csv(df_woout,"../CD_hourly_Data_10to21_WdSe_afterlinear_woOutlier.csv")
#write.csv(df_woout,"../HuCo_hourly_Data_10to21_WdSe_afterlinear_woOutlier.csv")

#歷年各天(365天)
d_yday_out = df_out %>% group_by(yday) %>% count()
d_yday_out$yday <- as.factor(d_yday_out$yday)
ggplot(d_yday_out) + geom_bar(aes(yday, n),stat='identity') + 
  labs(title='歷年各天(365天)_Outlier') + theme(plot.title = element_text(color=1, size=16, face="bold"))

# 各月(1~12) 
d_ym_out = df_out %>% group_by(month) %>% count()
d_ym_out$month <- as.factor(d_ym_out$month)
ggplot(d_ym_out) + geom_bar(aes(month, n),stat='identity') + labs(title='各月(1~12)_Outlier')

# 各週(週一、週二...)
d_wday_out = df_out %>% group_by(wday) %>% count()
d_wday_out$wday <- as.factor(d_wday_out$wday)
ggplot(d_wday_out) + geom_bar(aes(wday, n),stat='identity') + labs(title='各週(週一、週二...)_Outlier')


# 各小時(00~23)
d_hour_out = df_out %>% group_by(hour) %>% count()
d_hour_out$hour <- as.factor(d_hour_out$hour)
ggplot(d_hour_out) + geom_bar(aes(hour, n),stat='identity') + labs(title='各小時(00~23)_Outlier')