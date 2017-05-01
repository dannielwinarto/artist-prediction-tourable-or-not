rm(list = ls())
dev.off()
setwd("~/Documents/project3_FS_Live_tourable")

library(dplyr)
library(lubridate)
# library(ggplot2)
library(data.table)
# library(VIM)
library(bit64)
# library(outliers)
# library(stringr)
# library(tidyr)
# library(caret) # to create a fold 
library(randomForest)
library(xgboost)
setdiff = base::setdiff


artist_ticket_start_date = fread("artist_ticket_start_date.csv")
# glimpse(artist_ticket_start_date)
artist_ticket_start_date[, ticket_sale_start_date := ymd_hms(ticket_sale_start_date)][,artist_key := NULL] # changing colclass of ticket_sale_start_date messinto date 

labeled_data = fread("danniel_fs_live_04.csv")
labeled_data[, datestamp := ymd(datestamp)]
# table(labeled_data$artist_name)

# setdiff(unique(artist_ticket_start_date$artist_name), unique(labeled_data$artist_name)) # finding disrepancy artists among 2 dataset
# [1] "Jennxpenn"        "Mia and Adelaine" "DK4L"             "Tana Mongeau"    

data = merge(labeled_data, artist_ticket_start_date, by = "artist_name", all.x = T)
# table(data$artist_name)

# comparing Adelaine Morin vs Adelaine Morin case, we need to delete the later start_date
data = data[ !artist_name == "Adelaine Morin" ]
glimpse(data)
setkeyv(data, c("artist_name","datestamp"))

data.filtered = data[  datestamp <= ticket_sale_start_date ,.SD[seq( 1,.N)] ,by = artist_name]
data.filtered[,.N, by = artist_name]
# setdiff( unique(data$artist_name), unique(data.filtered$artist_name))



a = labeled_data[artist_name == "Vincent Cyr"]
View(a)
a = data[artist_name== "Alisha Marie"]

# Question:
# the following artist doesnt have needed records: "Alisha Marie"   "Greyson Chance" "Jack Douglass"  "Merrell Twins"  "Twaimz"         "Vincent Cyr"
# the artist doesnt have complete record:
# artist_name   N
# 1:       Adelaine Morrin  51
# 2:         Alexis G Zall  51
# 3:     AmandasChronicles 139
# 4:           Gabbie Show  65
# 5:           Mia Stammer  51
# 6: Niki & Gabi Demartino   5
# 7:       Savannah Soutas  29
 

glimpse(data.filtered)
data.filtered[, y := sapply(.SD[,y], as.factor)] # changing y column to be factor
data.filtered[, tag_id := NULL]

# getting the percentage change 
perc_chg = function(x){ # percentage change function
  temp  =(( shift(x, type = "lead")[1] - x[1])/x[1]*100)
  if(is.na(temp)){
    return(0)
  }
  return(temp)
}

percentage_change.ds = data.filtered[,lapply(.SD[c(1,.N)], perc_chg),  # finding the percent change among that 7 days, and then getting 0.5th  to 99.5 th percentile of the dataset to remove the outliers
                                     by = "artist_name", 
                                     .SDcols = c("views","subscribers", "videos")]
data.filtered[,.SD[1],by = artist_name][,c(1,4)]

setnames(percentage_change.ds, c("artist_name","perc_change_views","perc_change_subscribers","perc_change_videos"))


glimpse(data.filtered)
average.ds = data.filtered[, lapply(.SD, mean),by = c("artist_name", "y"), .SDcols = -c( "campaign_channel_key", "youtube_id", "artist_ucid", "datestamp","daynum","ticket_sale_start_date" )] # getting the performance average 
setnames(average.ds, c("artist_name", "y", paste0("avg_", names(average.ds)[c(-1,-2)])))

# joining the dataset
final.ds = merge(percentage_change.ds,average.ds, by = "artist_name")
setcolorder(final.ds, c(names(final.ds)[-5], "y")) # moving the dependent variable to the last column 

# table(final.ds$y)
# not_tourable     tourable 
# 1624           64

# a = data[y == "tourable" & !campaign_channel_key %in% tourable.ds$campaign_channel_key]
tourable.ds = final.ds[y == "tourable"]
non.tourable.ds = final.ds[y == "not_tourable"]

##########
# Building model 
##########

# Logistic Regression model
list_misclass_logistic = list()

for(i in 1:5){
  # due to lack of tourable data. the testing index consist of 35 percent tourable(22 obs) and 5 percent nontourable (81 obs). total is 103 observations
  # i = 1
  set.seed(i)
  testing.index.tourable = sample(tourable.ds$campaign_channel_key, size = .35*(nrow(tourable.ds)), replace = F)
  set.seed(i)
  testing.index.non.tourable = sample(non.tourable.ds$campaign_channel_key, size = .05*(nrow(non.tourable.ds)), replace = F)
  testing.data = rbindlist(list(
    final.ds[campaign_channel_key %in% testing.index.tourable], final.ds[campaign_channel_key %in% testing.index.non.tourable]
  ))
  # each training chunk, it consist of 42 obs tourable remaining and 100 nontourable obs. we will do ensamble of 14 iteration of bootstraping  sample of nontourable data 
  training.data = final.ds[!campaign_channel_key %in% testing.data$campaign_channel_key]
  training.data.tourable = training.data[ y == "tourable"]
  training.data.non.tourable = training.data[ y == "not_tourable"]
  
  logistic.prob.df = data.table() # this data.frame will store the probability outcome of logistic regression
  for (j in 1:17){
    # generating bootstrap sample (sampling with replacement)
    set.seed(j)
    training.data.non.tourable.sample = training.data.non.tourable[ campaign_channel_key %in% sample(training.data.non.tourable$campaign_channel_key, size = 100, replace = T)]
    training.data.sample = rbindlist(list(training.data.tourable,training.data.non.tourable.sample))
    logistic.model = glm(y ~. -campaign_channel_key, data = training.data.sample, family = "binomial")
    temp_name = paste0("logistic_model",j,".rds")
    saveRDS( logistic.model,file = temp_name)
    logistic.model.prob.sample = (predict(logistic.model, testing.data, type="response"))
    logistic.prob.df = rbind(logistic.prob.df, t(logistic.model.prob.sample) )
  }
  logistic.prob.average = apply(logistic.prob.df, 2, mean)
  logistic.pred = rep("not_tourable", nrow(testing.data))
  logistic.pred[logistic.prob.average> 0.5]="tourable"
  misclass_table_logistic  =  table(testing.data$y, logistic.pred)
  list_misclass_logistic[[i]] = misclass_table_logistic
} 

# Random Forest Model
list_misclass_RandomForest = list()
channel_id_FalseNegative = list()


for(i in 1:5){
  # due to lack of tourable data. the testing index consist of 35 percent tourable(22 obs) and 5 percent nontourable (81 obs). total is 103 observations
  # i = 1
  set.seed(i)
  testing.index.tourable = sample(tourable.ds$campaign_channel_key, size = .35*(nrow(tourable.ds)), replace = F)
  set.seed(i)
  testing.index.non.tourable = sample(non.tourable.ds$campaign_channel_key, size = .05*(nrow(non.tourable.ds)), replace = F)
  testing.data = rbindlist(list(
    final.ds[campaign_channel_key %in% testing.index.tourable], final.ds[campaign_channel_key %in% testing.index.non.tourable]
  ))
  # each training chunk, it consist of 42 obs tourable remaining and 100 nontourable obs. we will do ensamble of 14 iteration of bootstraping  sample of nontourable data 
  training.data = final.ds[!campaign_channel_key %in% testing.data$campaign_channel_key]
  training.data.tourable = training.data[ y == "tourable"]
  training.data.non.tourable = training.data[ y == "not_tourable"]
    
  # Random Forest Model 
  randomForest.pred.df = data.table()
  for (j in 1:17){
    # generating bootstrap sample (sampling with replacement)
    set.seed(j)
    training.data.non.tourable.sample = training.data.non.tourable[ campaign_channel_key %in% sample(training.data.non.tourable$campaign_channel_key, size = 100, replace = T)]
    training.data.sample = rbindlist(list(training.data.tourable,training.data.non.tourable.sample))
    randomForest.model = randomForest(y ~. -campaign_channel_key, 
                                      data = training.data.sample,
                                      mtry = 4, # for classification, the predictors mtry is sqrt(p)
                                      ntree = 500,
                                      importance = TRUE)
    randomForest.pred.sample = predict(randomForest.model, testing.data)
    randomForest.pred.df = rbind(randomForest.pred.df, as.factor(t(randomForest.pred.sample) ))
  }
  most_freq = function(x){ # finding the most frequent factor withn a column
    return(names(sort(table(x), decreasing = T))[1])
  }
  RF.pred = apply(randomForest.pred.df,2,most_freq)
  list_misclass_RandomForest[[i]] = table(testing.data$y,RF.pred)
  channel_ids_misclassified = testing.data$campaign_channel_key[1:length(testing.index.tourable)][testing.data[y == "tourable"]$y != RF.pred[1:length(testing.index.tourable)]]
  channel_id_FalseNegative[[i]] = testing.data[campaign_channel_key%in% channel_ids_misclassified]
}


#  XGBOOST Model


xgboost.pred.df = data.table()

set.seed(1)
testing.index.tourable = sample(tourable.ds$campaign_channel_key, size = .35*(nrow(tourable.ds)), replace = F)
set.seed(1)
testing.index.non.tourable = sample(non.tourable.ds$campaign_channel_key, size = .05*(nrow(non.tourable.ds)), replace = F)
testing.data = rbindlist(list(
  final.ds[campaign_channel_key %in% testing.index.tourable], final.ds[campaign_channel_key %in% testing.index.non.tourable]
))
test.x = data.matrix(testing.data[,-"y"])
test.y = testing.data$y
dtest = xgb.DMatrix(data =test.x, label = test.y)


# each training chunk, it consist of 42 obs tourable remaining and 100 nontourable obs. we will do ensamble of 14 iteration of bootstraping  sample of nontourable data 
training.data = final.ds[!campaign_channel_key %in% testing.data$campaign_channel_key]
training.data.tourable = training.data[ y == "tourable"]
training.data.non.tourable = training.data[ y == "not_tourable"]

xgb.prob.df = data.table() # this data.frame will store the probability outcome of logistic regression
for (j in 1:17){
  # generating bootstrap sample (sampling with replacement)
  set.seed(j)
  training.data.non.tourable.sample = training.data.non.tourable[ campaign_channel_key %in% sample(training.data.non.tourable$campaign_channel_key, size = 100, replace = T)]
  training.data.sample = rbindlist(list(training.data.tourable,training.data.non.tourable.sample))
  train.x = data.matrix(training.data.sample[,-"y"])
  train.y = as.numeric( training.data.sample$y) - 1 # 1 is tourable, zero is non tourable. XGboost logistic required the response to be 0 and 1
  dtrain = xgb.DMatrix(data = train.x, label = train.y)
  watchlist =  list(train=dtrain, test=dtest)
  set.seed(j)
  bst = xgb.train(data = dtrain,
                  booster = "gbtree", # tree based model
                  objective = "binary:logistic", # binary output classification
                  max.dept = 10, # depth of the tree
                  nthread = 2, # 2 core of CPU used
                  nround  = 20, 
                  eta = 1, 
                  watchlist =  watchlist,
                  eval.metric = "error")
  best_iter = bst$evaluation_log$iter[ min(bst$evaluation_log$train_error) ==  bst$evaluation_log$train_error][1] # the best iteration is when the training error is the mmin
  set.seed(j)
  xgb.model = xgboost(data = dtrain, 
                      booster = "gbtree",
                      objective = "binary:logistic",
                      max.dept = 10, # depth of the tree
                      nthread = 2, # 2 core of CPU used
                      nround  = best_iter, 
                      eta = 1)
  xgb.prob.sample = predict(xgb.model, dtest)
  xgb.prob.df = rbind(xgb.prob.df, t(xgb.prob.sample) )
}
xgb.prob.average = apply(xgb.prob.df, 2, mean)
xgb.pred = rep("not_tourable", nrow(testing.data))
xgb.pred[xgb.prob.average>0.5] = "tourable"
misclass_table_xgb  =  table(testing.data$y, xgb.pred)


final.ds[campaign_channel_key ==115]


The10Channels_example = fread("danniel_fs_live_10channels.csv")
The10Channels_example = The10Channels_example[, c("campaign_channel_key","youtube_id", "title")] [, youtube_link := paste0("https://www.youtube.com/channel/UC", youtube_id)][, youtube_id := NULL]
The10Channels_example = as.data.table(inner_join(The10Channels_example, final.ds, by ="campaign_channel_key" ))[, c("campaign_channel_key","youtube_link", "title","y" )]
The10Channels_example$logistic_pred = logistic.pred
The10Channels_example$RF.pred = RF.pred
The10Channels_example$xgb = xgb.pred




testing.data = final.ds[campaign_channel_key %in% The10Channels_example$campaign_channel_key]



training.data = final.ds[!campaign_channel_key %in% testing.data$campaign_channel_key]



write.csv(The10Channels_example, file = "example_implementation.csv")



  
  randomForest.model = randomForest(y ~. -campaign_channel_key, 
                                    data = training.data.sample,
                                    mtry = 4, # for classification, the predictors mtry is sqrt(p)
                                    ntree = 500,
                                    importance = TRUE)
  randomForest.pred.sample = predict(randomForest.model, testing.data)
  randomForest.pred.df = rbind(randomForest.pred.df, as.factor(t(randomForest.pred.sample) ))
}
most_freq = function(x){ # finding the most frequent factor withn a column
  return(names(sort(table(x), decreasing = T))[1])
}
RF.pred = apply(randomForest.pred.df,2,most_freq)
list_misclass_RandomForest[[i]] = table(testing.data$y,RF.pred)
channel_ids_misclassified = testing.data$campaign_channel_key[1:length(testing.index.tourable)][testing.data[y == "tourable"]$y != RF.pred[1:length(testing.index.tourable)]]
channel_id_FalseNegative[[i]] = testing.data[campaign_channel_key%in% channel_ids_misclassified]
}


for(i in seq(1,5)){
  two_wk_dt_train[two_wk_fold[[i]]]
  test.y = two_wk_dt_train[two_wk_fold[[i]]]$gross_revenue
  test.x = data.matrix( two_wk_dt_train[two_wk_fold[[i]]][, -c("gross_revenue")])
  train.y = two_wk_dt_train[-two_wk_fold[[i]]]$gross_revenue
  train.x = data.matrix( two_wk_dt_train[-two_wk_fold[[i]]][, -c("gross_revenue")])
  dtrain = xgb.DMatrix(data = train.x, label = train.y)
  dtest = xgb.DMatrix(data = test.x, label = test.y)
  watchlist =  list(train=dtrain, test=dtest)
  set.seed(1)
  bst = xgb.train(data = dtrain, 
                  objective = "reg:linear",
                  max.dept = 5, # depth of the tree
                  nthread = 2, # 2 core of CPU used
                  nround  = 20, 
                  eta = 1, 
                  watchlist =  watchlist)
  best_iter = grep(min(bst$evaluation_log$test_rmse),bst$evaluation_log$test_rmse )
  set.seed(1)
  xgb.model = xgboost(data = dtrain, 
                      objective = "reg:linear",
                      max.dept = 5, # depth of the tree
                      nthread = 2, # 2 core of CPU used
                      nround  = best_iter, 
                      eta = 1, 
                      watchlist)
  xgb.predict = predict(xgb.model, dtest)
  RMSE_xgb[i] = sqrt(mean((xgb.predict - test.y)^2))
}
end =  Sys.time()
duration.xgb = end - start
RMSE_xgb







hist(log(percentage_change.ds$views))
hist(((percentage_change.ds$subscribers + abs(min(percentage_change.ds$subscribers)+1))))

max(percentage_change.ds$videos)
min(rm.outlier(percentage_change.ds$subscribers))

hist(percentage_change.ds$views)
percentage_change.ds[(percentage_change.ds$views <0 )] 

setnames(percentage_change.ds,c("views","subscribers", "videos"),paste0("log_percent_change_",c("views","subscribers", "videos") ))



test = data[1:20]
a = test[,lapply(.SD[c(1,.N)], function(x){(100*(x-shift(x, type = "lag")))/x}), 
     by = "campaign_channel_key", 
     .SDcols = c("views","comments","subscribers", "videos", "videos_published_last_30days",
                 "videos_published_last_30days_views","videos_published_last_30days_likes",
                 "videos_published_last_30days_dislikes","videos_published_last_30days_comments")][,.SD[.N],by =  "campaign_channel_key"]
a[is.na(a)] = 0

#[,.SD[.N], by = "campaign_channel_key" ]
tbl_df(data)
a = seq(1:5)
shift(a, type = "lead")


data[,.SD[c(1,.N)], 
     by = "campaign_channel_key", 
     .SDcols = c("views","comments","subscribers", "videos", "videos_published_last_30days",
                 "videos_published_last_30days_views","videos_published_last_30days_likes",
                 "videos_published_last_30days_dislikes","videos_published_last_30days_comments")]
x = 1:5


percentage_change = function (x){
    return((x - shift(x))/x)  
}

is.nan(NaN)
  

  
(3063971- 3057592)/3063971

DT = data.table(year=2010:2014, v1=runif(5), v2=1:5, v3=letters[1:5])
cols = c("v1","v2","v3")
anscols = paste("lead", cols, sep="_")
DT[, (anscols) := shift(.SD, 1, 0, "lead"), .SDcols=cols]
myFunc <- function(x) x/shift(x)
a = c(1,2,3,4)
myFunc()


shift(a)

ashift(x)

?shift()

data[y == "not_tourable"]
a = data %>% filter(y == "not_tourable") %>% group_by(campaign_channel_key) %>% summarise(n())
a = data %>% filter(y == "tourable") %>% group_by(campaign_channel_key) %>% summarise(n())

glimpse(data)
# data = as.data.table(data)
hist(log10(data$views))
?log?
hist(rnorm(30))
max(data$views)
length(unique(data$campaign_channel_key))
data %>% group_by(campaign_channel_key) %>% summarise(a = n()) %>% arrange(-desc(a))
data %>% group_by(campaign_channel_key)

a = data %>% group_by(campaign_channel_key, y) %>% summarise( average_views = mean(views))

not_t = a[a$y == "not_tourable",]
yes_t = a[a$y == "tourable",]
summary(not_t$average_views)
hist(log10(not_t$average_views))
hist(log10(yes_t$average_views))

a = data[views >= 1000000000 ,]
length(unique(a$campaign_channel_key))
hist(data[])
table(data$y)

sum(data$)

aggr(data) #  there are no missing data








