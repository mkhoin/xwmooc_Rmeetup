#install.packages(c('MLmetrics', 'rjson', 'tidyverse', 'knitr', 'purrr', 'stringr', 'lubridate', 'rJava', 'sqldf', 'modelr', 'ModelMetrics', 'caret'))
#install.packages('xgboost')

#윈도우 경우
#1. mingw-w64 다운로드 및 설치
#https://sourceforge.net/projects/mingw-w64/
  
#2.윈도우에 XGBoost 설치하기
#http://quantfactory.blogspot.kr/2017/04/xgboost.html

require(rjson)
require(dplyr)
require(purrr)
require(knitr)
require(stringr)
require(lubridate)
require(tidyr)
require(rJava)
require(sqldf)
require(modelr)
require(ModelMetrics)
require(caret)

require(xgboost)


setwd('/Users/syleeie/Desktop/xwmooc_Rmeetup/Meetup_1/modelr')
options(tibble.width = Inf)

train = fromJSON(file = "./data/train.json")
test = fromJSON(file = "./data/test.json")

column <- setdiff(names(train), c("photos", "features"))
column

train <- map_at(train, column, unlist) %>% tibble::as_tibble(.)
test <- map_at(test, column, unlist) %>% tibble::as_tibble(.)

colnames(train)
glimpse(train)

train <- train %>% mutate(
                  bathrooms = as.numeric(bathrooms),
                  bedrooms = as.numeric(bedrooms),
                  created = ymd_hms(created),
                  latitude = as.numeric(latitude),
                  longitude = as.numeric(longitude),
                  price = as.numeric(price),
                  created_week = weeks(created),
                  created_hour = hour(created),
                  created_weekday = wday(created),
                  created_month = month(created),
                  price_t = price / bedrooms,
                  feature_count = lengths(features),
                  photo_count = lengths(photos),
                  description_count = lengths(str_split(description, "\\s+")),
                  interest_level = factor(interest_level, levels=c('low', 'medium', 'high')))

glimpse(train)

test <- test %>% mutate(bathrooms = as.numeric(bathrooms),
                          bedrooms = as.numeric(bedrooms),
                          created = ymd_hms(created),
                          latitude = as.numeric(latitude),
                          longitude = as.numeric(longitude),
                          price = as.numeric(price),
                          created_week = weeks(created),
                          created_hour = hour(created),
                          created_weekday = wday(created),
                          created_month = month(created),
                          price_t = price / bedrooms,
                          feature_count = lengths(features),
                          photo_count = lengths(photos),
                          description_count = lengths(str_split(description, "\\s+"))
                          )
glimpse(test)

features_to_use = c("bathrooms", "bedrooms", "latitude", "longitude", "price", "price_t", "photo_count", "feature_count", "description_count", "created_week", "created_hour", "created_weekday", "created_month", "listing_id")

features_to_use
glimpse(train)  

manager_count <- train %>% 
          group_by(manager_id, interest_level) %>%
          summarise(
            count = n()
          )

manager_count[18:19,]

manager_ratio <- sqldf("
        select manager_id, (low_count*1.0)/sum_count low_ratio
                           , (medium_count*1.0)/sum_count medium_ratio 
                           , (high_count*1.0)/sum_count high_ratio
                           , sum_count
        from (
        select manager_id, 
        sum(case when interest_level == 'low' then count*1.0 else 0 end) as low_count,
        sum(case when interest_level == 'medium' then count*1.0 else 0 end) as medium_count, 
        sum(case when interest_level == 'high' then count*1.0 else 0 end) as high_count, 
        sum(count) sum_count
        from manager_count
        group by manager_id
        )t")

manager_ratio %>% filter(manager_id == '77f81a0a8af6db8349587acefd1b533f')

train <- train %>% left_join(manager_ratio, by = 'manager_id')
test <- test %>% left_join(manager_ratio, by = 'manager_id')

train <- train %>% mutate(
  display_address = as.integer(as.factor(display_address)),
  manager_id = as.integer(as.factor(manager_id)),
  building_id = as.integer(as.factor(building_id)),
  street_address = as.integer(as.factor(street_address)),
)

test <- test %>% mutate(
  display_address = as.integer(as.factor(display_address)),
  manager_id = as.integer(as.factor(manager_id)),
  building_id = as.integer(as.factor(building_id)),
  street_address = as.integer(as.factor(street_address)),
)

features_to_use = c(features_to_use,'low_ratio', 'medium_ratio', 'high_ratio', 'sum_count', 'display_address', 'manager_id', 'building_id', 'street_address')
features_to_use

#https://flonelin.wordpress.com/2016/07/26/tuning-xgboostextream-gradient-boosting/

runXGB <- function(train_X, train_y, test_X, test_y=NULL, feature_names=NULL, seed_val=0, num_rounds=100){
    param <- list()
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.03
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7 # 70프로만 뽑겠다
    param['colsample_bytree'] = 0.7 # 변수 컬럼 비율
    param['seed'] = seed_val # 초기값 설정 랜덤 안되게
    num_rounds = num_rounds

    plst = param
    xgtrain = xgb.DMatrix(data=train_X, label=train_y) 
    feature_names = dimnames(xgtrain)

    # test의 타겟값을 넣은 xgb model
    if (!is.null(test_y)){
      xgtest = xgb.DMatrix(data=test_X, label=test_y)
      watchlist = list('train'=xgtrain, 'test'=xgtest)
      model = xgb.train(params=plst, data=xgtrain, nrounds=num_rounds, watchlist=watchlist, early_stopping_rounds=20)
    }
    else{
      xgtest = xgb.DMatrix(test_X)
      model = xgb.train(plst, xgtrain, num_rounds)
    }
    pred_test_y = predict(model, xgtest)
    return(list(pred_test_y, model, feature_names))
}

cv_scores = list()
train_df <- train[features_to_use]
train_y <- train['interest_level'] %>% mutate(interest_level = as.numeric(interest_level)-1)
kf <- crossv_kfold(train_df, 5)

kf[[1]][1]
kf[[2]][1]

for (i in 1:5){
  dev_x <- as.matrix(train_df[kf$train[[i]]$idx,])
  val_x <- as.matrix(train_df[kf$test[[i]]$idx,])
  dev_y <- as.matrix(train_y[kf$train[[i]]$idx,])
  val_y <- as.matrix(train_y[kf$test[[i]]$idx,])
  results <- runXGB(dev_x, dev_y, val_x, val_y)
  cv_scores <- append(cv_scores, mlogLoss(actual = as.factor(val_y),
                                          predicted = matrix(results[[1]],ncol=3, byrow=T)))
  print(cv_scores)
  break
}

#caret 사용시
kf <- createFolds(1:dim(train_y)[1], k = 5)
str(kf)

#modelr crossv_mc 사용시 
#Monte Carlo cross-validation
kf <- crossv_mc(train_df, 5)

for (i in 1:5){
  dev_x <- as.matrix(train_df[kf$train[[i]]$idx,])
  val_x <- as.matrix(train_df[kf$test[[i]]$idx,])
  dev_y <- as.matrix(train_y[kf$train[[i]]$idx,])
  val_y <- as.matrix(train_y[kf$test[[i]]$idx,])
  results <- runXGB(dev_x, dev_y, val_x, val_y)
  cv_scores <- append(cv_scores, mlogLoss(actual = as.factor(val_y),
                                          predicted = matrix(results[[1]],ncol=3, byrow=T)))
  print(cv_scores)
  break
}


#submission
test_df <- test[features_to_use]
result_test = runXGB(as.matrix(train_df), as.matrix(train_y), as.matrix(test_df), num_rounds=100)
out_df = as.data.frame(matrix(result_test[[1]], ncol=3, byrow=T))
colnames(out_df) = c("low", "medium", "high")
out_df['listing_id'] <- test_df %>% select(listing_id)
write.csv(out_df, file='data/xgb_starter_num_rounds100_xwmooc_1meetup.csv', row.names = FALSE)

