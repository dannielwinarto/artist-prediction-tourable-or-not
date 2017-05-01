#!/usr/bin/env ruby

# START DATE : Mar 17, 2017
# COMPANY : 
# DUDE: Dr. Vincent Seah
# LAST UPDATE: Mar 17, 2017
# VERSION : 0.1



require 'rubygems'
require 'matrix'
require 'json'
require 'csv'
require 'rinruby'



class Danniel
  
  
  def test(input)
    currentdir = File.dirname(__FILE__)
    # filename = mydata
    puts input

    # CSV.foreach("#{currentdir}/testing_input2.csv") do |row|
    #     p row 

    # end
    r_env = RinRuby.new(echo = true)


    # arr_of_arrs = CSV.read("#{currentdir}/testing_input2.csv")
    # p arr_of_arrs

r_env.eval <<EOF
    library(data.table)
    library(lubridate)
    library(dplyr)
    data = fread("danniel_tourable_data_v2.csv")
    data$views  = as.numeric(data$views)
    data[, y := sapply(.SD[,y], as.factor)] # changing y column to be factor
    data[, tag_id := NULL]
    data$datestamp = ymd(data[,datestamp]) # converting to date
    temp = data[,.N, by ="campaign_channel_key"][ N >= 7][,1] # finding the list of channel
    data = data[campaign_channel_key %in% temp$campaign_channel_key] # this is the data with 7 or more records
    setkeyv(data, c("campaign_channel_key", "datestamp"))
    perc_chg = function(x){ # percentage change function
        temp  =(( shift(x, type = "lead")[1] - x[1])/x[1]*100)
        if(is.na(temp)){
            return(0)
        }
        return(temp)
    }
    percentage_change.ds = data[,lapply(.SD[c(1,.N)], perc_chg),  # finding the percent change among that 7 days, and then getting 0.5th  to 99.5 th percentile of the dataset to remove the outliers
                                by = "campaign_channel_key", 
                                .SDcols = c("views","subscribers", "videos")][, .SD[(views <= quantile(views, probs = .995)  & views >= quantile(views, probs = .005)) &
                                        (subscribers <= quantile(subscribers, probs = .995)  & subscribers >= quantile(subscribers, probs = .005)) &
                                        (videos <= quantile(videos, probs = .995)  & videos >= quantile(videos, probs = .005)) ]]
    setnames(percentage_change.ds, c("campaign_channel_key","perc_change_views","perc_change_subscribers","perc_change_videos"))
    average.ds = data[, lapply(.SD, mean),by = c("campaign_channel_key", "y"), .SDcols = -c( "datestamp","daynum")] # getting the performance average 
    setnames(average.ds, c("campaign_channel_key", "y", paste0("avg_", names(average.ds)[c(-1,-2)])))
    final.ds = as.data.table(inner_join(percentage_change.ds,average.ds, by = "campaign_channel_key"))
    setcolorder(final.ds, c(names(final.ds)[-5], "y")) # moving the dependent variable to the last column 
    logistic_model_1 = readRDS("logistic_model1.rds")
    logistic_model_2 = readRDS("logistic_model2.rds")
    logistic_model_3 = readRDS("logistic_model3.rds")
    logistic_model_4 = readRDS("logistic_model4.rds")
    logistic_model_5 = readRDS("logistic_model5.rds")
    logistic_model_6 = readRDS("logistic_model6.rds")
    logistic_model_7 = readRDS("logistic_model7.rds")
    logistic_model_8 = readRDS("logistic_model8.rds")
    logistic_model_9 = readRDS("logistic_model9.rds")
    logistic_model_10 = readRDS("logistic_model10.rds")



    
    
    logistic.prob.df = data.table() # this data.frame will store the probability outcome of logistic regression

    logistic.model.prob_1 = (predict(logistic_model_1, final.ds , type="response"))
    logistic.model.prob_2 = (predict(logistic_model_2, final.ds , type="response"))
    logistic.model.prob_3 = (predict(logistic_model_3, final.ds , type="response"))
    logistic.model.prob_4 = (predict(logistic_model_4, final.ds , type="response"))
    logistic.model.prob_5 = (predict(logistic_model_5, final.ds , type="response"))
    logistic.model.prob_6 = (predict(logistic_model_6, final.ds , type="response"))
    logistic.model.prob_7 = (predict(logistic_model_7, final.ds , type="response"))
    logistic.model.prob_8 = (predict(logistic_model_8, final.ds , type="response"))
    logistic.model.prob_9 = (predict(logistic_model_9, final.ds , type="response"))
    logistic.model.prob_10 = (predict(logistic_model_10, final.ds , type="response"))
    

    logistic.prob.df = rbind( logistic.model.prob_1, logistic.model.prob_2, logistic.model.prob_3,logistic.model.prob_4,logistic.model.prob_5,logistic.model.prob_6,
                                        logistic.model.prob_7,logistic.model.prob_8,logistic.model.prob_9,logistic.model.prob_10)
    # logistic.prob.df
    aggregated.prob = apply(logistic.prob.df, 2, mean) 
    aggregated.prob.df = data.frame(aggregated.prob)
    aggregated_prob_df <- (aggregated.prob.df)


    # logistic.pred = rep("not_tourable", nrow(final.ds))
    # logistic.pred[aggregated.prob>=.7]="tourable"
    # logistic.pred
    # table(logistic.pred)
EOF
    puts r_env.aggregated_prob_df.to_a
  end
end

Danniel.new().test('hello')