xwMOOC R Meetup 1회차
========================================================
author: Sang Yeol Lee
date: August 23 2017
width: 1500 
height: 1800
transition: linear
transition-speed: slow
autosize: true

<h3>
- 대상자 : R를 사용하고 좋아하는 참가자 누구나
- 주제 : Tidyverse - modelr

- 운영 지원: [캐글뽀개기](https://www.facebook.com/groups/kagglebreak) 

- 장소 지원: [kosslab](https://kosslab.kr/)


- [페북 이벤트 링크](https://www.facebook.com/events/1822675967748595)

- [dropbox 자료 다운로드 링크](https://www.dropbox.com/sh/0n3650m6ifpdjxn/AAArgiJz5LnNdVTNUhpxjpGza?dl=0)

========================================================
id: slide1
type: prompt

# Tidyverse -- Modelr
  <h2>
  - * modelr 기능 소개 (2/6h)
  - * 캐글 문제 소개 및 적용 (4/6h)
  - * 회귀분석 문제 적용 (생략)
  - * Q&A 

<br>

## 자료참고 레퍼런스
  [R4DS](http://r4ds.had.co.nz/1)
  
  [Tidyverse](https://www.tidyverse.org/packages/)
  
  [Pyconkr 2017 kaggle tutorial](https://github.com/KaggleBreak/walkingkaggle/blob/master/pycon2017_kr/pycon_korea_2017_Kaggle_tutorial.ipynb)

<br>

========================================================
## 1단계 : 라이브러리 소개 (modelr)
[Go to slide 1](#/slide1)

- Modelling Functions that Work with the Pipe

![Index.html](./img/img1.png)

- 원활한 통합을 지원하는 모델링을 위한 함수
데이터 조작 및 시각화의 파이프 라인으로 모델링

========================================================

## 1단계 라이브러리 설치 및 기본기능 소개

### Partitioning and sampling


```r
#install.packages('tidyverse', dependencies = T, repos='http://cran.r-project.org')
#install.packages('modelr', dependencies = T, repos='http://cran.r-project.org')

library(tidyverse)
library(modelr)

# a subsample of the first ten rows in the data frame
rs <- resample(mtcars, 1:10)
as.data.frame(rs)
```

```
                   mpg cyl  disp  hp drat    wt  qsec vs am gear carb
Mazda RX4         21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
Mazda RX4 Wag     21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
Datsun 710        22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
Hornet 4 Drive    21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
Hornet Sportabout 18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
Valiant           18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
Duster 360        14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
Merc 240D         24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
Merc 230          22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
Merc 280          19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
```

```r
rs2 <- resample(mtcars, 1:10)
as.data.frame(rs2)
```

```
                   mpg cyl  disp  hp drat    wt  qsec vs am gear carb
Mazda RX4         21.0   6 160.0 110 3.90 2.620 16.46  0  1    4    4
Mazda RX4 Wag     21.0   6 160.0 110 3.90 2.875 17.02  0  1    4    4
Datsun 710        22.8   4 108.0  93 3.85 2.320 18.61  1  1    4    1
Hornet 4 Drive    21.4   6 258.0 110 3.08 3.215 19.44  1  0    3    1
Hornet Sportabout 18.7   8 360.0 175 3.15 3.440 17.02  0  0    3    2
Valiant           18.1   6 225.0 105 2.76 3.460 20.22  1  0    3    1
Duster 360        14.3   8 360.0 245 3.21 3.570 15.84  0  0    3    4
Merc 240D         24.4   4 146.7  62 3.69 3.190 20.00  1  0    4    2
Merc 230          22.8   4 140.8  95 3.92 3.150 22.90  1  0    4    2
Merc 280          19.2   6 167.6 123 3.92 3.440 18.30  1  0    4    4
```

==================
## 1단계 기본기능 소개
### Partitioning and sampling


```r
ex <- resample_partition(mtcars, c(test = 0.3, train = 0.7))
ex
```

```
$test
<resample [9 x 11]> 1, 4, 5, 9, 15, 19, 22, 24, 27

$train
<resample [23 x 11]> 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, ...
```

```r
lapply(ex, dim)
```

```
$test
[1]  9 11

$train
[1] 23 11
```

==================
## 1단계 기본기능 소개
### Partitioning and sampling


```r
# bootstrap
boot <- bootstrap(mtcars, 100)
boot
```

```
# A tibble: 100 x 2
            strap   .id
           <list> <chr>
 1 <S3: resample>   001
 2 <S3: resample>   002
 3 <S3: resample>   003
 4 <S3: resample>   004
 5 <S3: resample>   005
 6 <S3: resample>   006
 7 <S3: resample>   007
 8 <S3: resample>   008
 9 <S3: resample>   009
10 <S3: resample>   010
# ... with 90 more rows
```

```r
dim(mtcars)
```

```
[1] 32 11
```

```r
boot$strap[[1]]
```

```
<resample [32 x 11]> 5, 25, 17, 6, 16, 17, 32, 18, 5, 27, ...
```

```r
dim(boot$strap[[1]])
```

```
[1] 32 11
```

```r
# k-fold cross-validation
cv1 <- crossv_kfold(mtcars, 5)
cv1
```

```
# A tibble: 5 x 3
           train           test   .id
          <list>         <list> <chr>
1 <S3: resample> <S3: resample>     1
2 <S3: resample> <S3: resample>     2
3 <S3: resample> <S3: resample>     3
4 <S3: resample> <S3: resample>     4
5 <S3: resample> <S3: resample>     5
```

```r
dim(cv1$train[[1]])
```

```
[1] 25 11
```

```r
dim(cv1$test[[1]])
```

```
[1]  7 11
```

==================
## 1단계 기본기능 소개
### Partitioning and sampling


```r
# Monte Carlo cross-validation
cv2 <- crossv_mc(mtcars, 100)
cv2
```

```
# A tibble: 100 x 3
            train           test   .id
           <list>         <list> <chr>
 1 <S3: resample> <S3: resample>   001
 2 <S3: resample> <S3: resample>   002
 3 <S3: resample> <S3: resample>   003
 4 <S3: resample> <S3: resample>   004
 5 <S3: resample> <S3: resample>   005
 6 <S3: resample> <S3: resample>   006
 7 <S3: resample> <S3: resample>   007
 8 <S3: resample> <S3: resample>   008
 9 <S3: resample> <S3: resample>   009
10 <S3: resample> <S3: resample>   010
# ... with 90 more rows
```

```r
dim(cv2$train[[1]])
```

```
[1] 25 11
```

```r
dim(cv2$test[[1]])
```

```
[1]  7 11
```

==================
## 1단계 기본기능 소개
### Model quality metrics


```r
#modelr includes several often-used model quality metrics:
mod <- lm(mpg ~ wt, data = mtcars) #mpg : Miles/US gallon, wt (weight)
rmse(mod, mtcars) #root-mean-squared-error
```

```
[1] 2.949163
```

```r
rsquare(mod, mtcars) #rsquare is the variance of the predictions divided by by the variance of the response
```

```
[1] 0.7528328
```

```r
mae(mod, mtcars) #the mean absolute error
```

```
[1] 2.340642
```

```r
qae(mod, mtcars) #quantiles of absolute error
```

```
       5%       25%       50%       75%       95% 
0.1784985 1.0005640 2.0946199 3.2696108 6.1794815 
```

==================
## 1단계 기본기능 소개
### Interacting with models


```r
#A set of functions let you seamlessly add predictions and residuals as additional columns to an existing data frame:
df <- tibble::data_frame(
  x = sort(runif(100)),
  y = 5 * x + 0.5 * x ^ 2 + 3 + rnorm(length(x))
)

mod <- lm(y ~ x, data = df)
df %>% add_predictions(mod)
```

```
# A tibble: 100 x 3
            x        y     pred
        <dbl>    <dbl>    <dbl>
 1 0.01987267 1.548814 2.805815
 2 0.02535856 2.684403 2.838197
 3 0.03073210 2.964052 2.869915
 4 0.03320441 3.863466 2.884509
 5 0.04125782 3.156826 2.932046
 6 0.04668376 4.738901 2.964074
 7 0.05863389 3.407387 3.034613
 8 0.06395357 3.854999 3.066014
 9 0.06850298 4.612543 3.092868
10 0.07344073 2.664695 3.122014
# ... with 90 more rows
```

```r
df %>% add_residuals(mod)
```

```
# A tibble: 100 x 3
            x        y      resid
        <dbl>    <dbl>      <dbl>
 1 0.01987267 1.548814 -1.2570012
 2 0.02535856 2.684403 -0.1537938
 3 0.03073210 2.964052  0.0941362
 4 0.03320441 3.863466  0.9789573
 5 0.04125782 3.156826  0.2247794
 6 0.04668376 4.738901  1.7748264
 7 0.05863389 3.407387  0.3727741
 8 0.06395357 3.854999  0.7889850
 9 0.06850298 4.612543  1.5196753
10 0.07344073 2.664695 -0.4573197
# ... with 90 more rows
```

==================
## 1단계 기본기능 소개
### Interacting with models


```r
#For visualization purposes it is often useful to use an evenly spaced grid of points from the data:
data_grid(mtcars, wt = seq_range(wt, 10), cyl, vs)
```

```
# A tibble: 60 x 3
         wt   cyl    vs
      <dbl> <dbl> <dbl>
 1 1.513000     4     0
 2 1.513000     4     1
 3 1.513000     6     0
 4 1.513000     6     1
 5 1.513000     8     0
 6 1.513000     8     1
 7 1.947556     4     0
 8 1.947556     4     1
 9 1.947556     6     0
10 1.947556     6     1
# ... with 50 more rows
```

```r
# For continuous variables, seq_range is useful
mtcars_mod <- lm(mpg ~ wt + cyl + vs, data = mtcars)
data_grid(mtcars, wt = seq_range(wt, 10), cyl, vs) %>% add_predictions(mtcars_mod)
```

```
# A tibble: 60 x 4
         wt   cyl    vs     pred
      <dbl> <dbl> <dbl>    <dbl>
 1 1.513000     4     0 28.37790
 2 1.513000     4     1 28.90207
 3 1.513000     6     0 25.64969
 4 1.513000     6     1 26.17386
 5 1.513000     8     0 22.92148
 6 1.513000     8     1 23.44566
 7 1.947556     4     0 26.96717
 8 1.947556     4     1 27.49134
 9 1.947556     6     0 24.23896
10 1.947556     6     1 24.76314
# ... with 50 more rows
```


========================================================
## 2단계 : 캐글 소개 및 적용
[Go to slide 1](#/slide1)

- [Kaggle](https://www.kaggle.com/)

  - Kaggle은 기업 및 연구원이 데이터 및 통계를 게시하고 데이터 마이너가 예측 및 설명을 위한 최상의 모델을 만들기 위해 경쟁하는 예측 모델링 및 분석 경쟁을위한 플랫폼. 

  - Crowdsourcing 접근 방식은 예측 모델링 작업에 적용 할 수있는 무수한 전략이 있으며 어떤 기술 또는 분석가가 가장 효과적인지를 처음부터 알 수 없다는 사실에 의존.

<br>

- [뉴욕시 임대아파트 문제](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) 

  - RentHop은 데이터를 사용하여 임대 목록을 품질 별로 분류하여 아파트 검색을 더 똑똑하게 만드는 사이트
  
  - 완벽한 아파트를 찾긴 어렵지만 사용 가능한 모든 부동산 데이터를 프로그래밍 방식으로 구조화하고 이해하는 것이 더욱 어려움.
  
![RentHop](https://camo.githubusercontent.com/92bded67b056d8f78a0121e8b17e39777a100b82/68747470733a2f2f6b6167676c65322e626c6f622e636f72652e77696e646f77732e6e65742f636f6d7065746974696f6e732f6b6167676c652f353539302f6d656469612f74776f7369676d612d72656e74686f702d62616e6e65722d323530783235302e706e67)


- [Renthop](https://www.renthop.com/)

  - RentHop은 사용자가 뉴욕, 보스턴, 시카고 및 다른 대도시 지역의 아파트를 검색 할 수있게 해주는 웹 및 모바일 기반 검색 엔진. 이 사이트는 실시간 리스팅을 제공하며 품질에 따라 아파트를 분류하는 고유 한 정렬 알고리즘을 사용
  
![img](https://github.com/KaggleBreak/walkingkaggle/raw/bb5aefa72dd3e930ad618f8a733836c507353edd/pycon2017_kr/img/renthop_1.png)

==================
## 2단계 캐글 소개 및 적용

### 목표 : modelr 기능 일부를 캐글에 사용하자. (귀찮은 CV...)

<br>

## 데이터 소개

### File descriptions
- train.json - the training set
- test.json - the test set
- sample_submission.csv - a sample submission file in the correct format
- images_sample.zip - listing images organized by listing_id (a sample of 100 listings)
- Kaggle-renthop.7z - (optional) listing images organized by listing_id. Total size: 78.5GB compressed. Distributed by BitTorrent (Kaggle-renthop.torrent).

<br>

### Data fields
- bathrooms: number of bathrooms
- bedrooms: number of bathrooms
- listing_id: 포스팅 ID
- building_id: 건물 ID
- manager_id: 포스팅 게시자 ID
- created: 포스팅 된 시각 (UTC 기준으로 생각됨)
- latitude: 위도
- longitude: 경도
- price: in USD
- features: a list of features about this apartment
- photos: a list of photo links. You are welcome to download the pictures yourselves from renthop's site, but they are the same as imgs.zip.
- display_address: 주소 (도로명까지)
- street_address: 주소 (번지까지) 

### - interest_level: this is the target variable. It has 3 categories: 'high', 'medium', 'low'

[데이터 다운로드 링크](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data)

- 데이터 다운로드 받으려면 캐글 가입 및 대회 동의를 해야함. 가입할 때 인증이 필요


==================
## 2단계 캐글 소개 및 적용


## 윈도우 경우

### [1. rJava  설치](https://thebook.io/006723/ch09/02/03/02-2/)

### 2 .xgboost 설치
[1. mingw-w64 다운로드 및 설치](https://sourceforge.net/projects/mingw-w64/)
  
[2.윈도우에 XGBoost 설치하기](http://quantfactory.blogspot.kr/2017/04/xgboost.html)


```r
#install.packages(c('MLmetrics', 'rjson', 'tidyverse', 'knitr', 'purrr', 'stringr', 'lubridate', 'rJava', 'sqldf', 'modelr', 'ModelMetrics', 'caret'))
#install.packages('xgboost')

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

options(tibble.width = Inf) #tibble 출력 컬럼 변경
```

==================
## 2단계 캐글 소개 및 적용

### 데이터 로딩

```r
train = fromJSON(file = "./data/train.json")
test = fromJSON(file = "./data/test.json")

column <- setdiff(names(train), c("photos", "features")) #photos, features는 리스트 형태로 저장하기 위해서
column
```

```
 [1] "bathrooms"       "bedrooms"        "building_id"    
 [4] "created"         "description"     "display_address"
 [7] "latitude"        "listing_id"      "longitude"      
[10] "manager_id"      "price"           "street_address" 
[13] "interest_level" 
```

```r
train <- map_at(train, column, unlist) %>% tibble::as_tibble(.)

test <- map_at(test, column, unlist) %>% tibble::as_tibble(.)

colnames(train)
```

```
 [1] "bathrooms"       "bedrooms"        "building_id"    
 [4] "created"         "description"     "display_address"
 [7] "features"        "latitude"        "listing_id"     
[10] "longitude"       "manager_id"      "photos"         
[13] "price"           "street_address"  "interest_level" 
```

```r
glimpse(train) #tibble 형태에서 str과 유사한 기능
```

```
Observations: 49,352
Variables: 15
$ bathrooms       <dbl> 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 2.0, 1.0, 0.0, 3...
$ bedrooms        <dbl> 1, 2, 2, 3, 0, 3, 3, 0, 1, 3, 0, 2, 0, 1, 2, 0...
$ building_id     <chr> "8579a0b0d54db803821a35a4a615e97a", "b8e75fc94...
$ created         <chr> "2016-06-16 05:55:27", "2016-06-01 05:44:33", ...
$ description     <chr> "Spacious 1 Bedroom 1 Bathroom in Williamsburg...
$ display_address <chr> "145 Borinquen Place", "East 44th", "East 56th...
$ features        <list> [<"Dining Room", "Pre-War", "Laundry in Build...
$ latitude        <dbl> 40.7108, 40.7513, 40.7575, 40.7145, 40.7439, 4...
$ listing_id      <dbl> 7170325, 7092344, 7158677, 7211212, 7225292, 7...
$ longitude       <dbl> -73.9539, -73.9722, -73.9625, -73.9425, -73.97...
$ manager_id      <chr> "a10db4590843d78c784171a107bdacb4", "955db3347...
$ photos          <list> [<"https://photos.renthop.com/2/7170325_3bb5a...
$ price           <dbl> 2400, 3800, 3495, 3000, 2795, 7200, 6000, 1945...
$ street_address  <chr> "145 Borinquen Place", "230 East 44th", "405 E...
$ interest_level  <chr> "medium", "low", "medium", "medium", "low", "l...
```

==================
## 2단계 캐글 소개 및 적용

### 데이터 전처리 및 변수 추가


```r
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
features_to_use = c("bathrooms", "bedrooms", "latitude", "longitude", "price", "price_t", "photo_count", "feature_count", "description_count", "created_week", "created_hour", "created_weekday", "created_month", "listing_id")

features_to_use
```

```
 [1] "bathrooms"         "bedrooms"          "latitude"         
 [4] "longitude"         "price"             "price_t"          
 [7] "photo_count"       "feature_count"     "description_count"
[10] "created_week"      "created_hour"      "created_weekday"  
[13] "created_month"     "listing_id"       
```

```r
glimpse(train) 
```

```
Observations: 49,352
Variables: 23
$ bathrooms         <dbl> 1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 2.0, 1.0, 0.0,...
$ bedrooms          <dbl> 1, 2, 2, 3, 0, 3, 3, 0, 1, 3, 0, 2, 0, 1, 2,...
$ building_id       <chr> "8579a0b0d54db803821a35a4a615e97a", "b8e75fc...
$ created           <dttm> 2016-06-16 05:55:27, 2016-06-01 05:44:33, 2...
$ description       <chr> "Spacious 1 Bedroom 1 Bathroom in Williamsbu...
$ display_address   <chr> "145 Borinquen Place", "East 44th", "East 56...
$ features          <list> [<"Dining Room", "Pre-War", "Laundry in Bui...
$ latitude          <dbl> 40.7108, 40.7513, 40.7575, 40.7145, 40.7439,...
$ listing_id        <dbl> 7170325, 7092344, 7158677, 7211212, 7225292,...
$ longitude         <dbl> -73.9539, -73.9722, -73.9625, -73.9425, -73....
$ manager_id        <chr> "a10db4590843d78c784171a107bdacb4", "955db33...
$ photos            <list> [<"https://photos.renthop.com/2/7170325_3bb...
$ price             <dbl> 2400, 3800, 3495, 3000, 2795, 7200, 6000, 19...
$ street_address    <chr> "145 Borinquen Place", "230 East 44th", "405...
$ interest_level    <fctr> medium, low, medium, medium, low, low, low,...
$ created_week      <S4: Period> 10262395689d 0H 0M 0S, 10253319111d 0...
$ created_hour      <int> 5, 5, 15, 7, 3, 5, 6, 5, 4, 3, 2, 2, 5, 3, 4...
$ created_weekday   <dbl> 5, 4, 3, 6, 3, 3, 4, 1, 5, 3, 7, 4, 7, 7, 7,...
$ created_month     <dbl> 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,...
$ price_t           <dbl> 2400.000, 1900.000, 1747.500, 1000.000, Inf,...
$ feature_count     <int> 7, 6, 6, 0, 4, 6, 5, 5, 1, 2, 5, 12, 11, 1, ...
$ photo_count       <int> 12, 6, 6, 5, 4, 5, 7, 5, 4, 11, 4, 5, 4, 6, ...
$ description_count <int> 76, 130, 118, 94, 40, 129, 70, 61, 53, 197, ...
```

==================
## 2단계 캐글 소개 및 적용

### 데이터 전처리 및 변수 추가


```r
manager_count <- train %>% 
          group_by(manager_id, interest_level) %>%
          summarise(
            count = n()
          )

manager_count[18:19,]
```

```
# A tibble: 2 x 3
# Groups:   manager_id [1]
                        manager_id interest_level count
                             <chr>         <fctr> <int>
1 016ae4f8903a08719d9d9f232d61d3ba            low     3
2 016ae4f8903a08719d9d9f232d61d3ba           high     3
```

```r
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
```

```
                        manager_id low_ratio medium_ratio high_ratio
1 77f81a0a8af6db8349587acefd1b533f 0.6436782    0.2758621 0.08045977
  sum_count
1        87
```

```r
train <- train %>% left_join(manager_ratio, by = 'manager_id')
test <- test %>% left_join(manager_ratio, by = 'manager_id')
```

==================
## 2단계 캐글 소개 및 적용

### 데이터 전처리 및 변수 추가


```r
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
```

```
 [1] "bathrooms"         "bedrooms"          "latitude"         
 [4] "longitude"         "price"             "price_t"          
 [7] "photo_count"       "feature_count"     "description_count"
[10] "created_week"      "created_hour"      "created_weekday"  
[13] "created_month"     "listing_id"        "low_ratio"        
[16] "medium_ratio"      "high_ratio"        "sum_count"        
[19] "display_address"   "manager_id"        "building_id"      
[22] "street_address"   
```


==================
## 2단계 캐글 소개 및 적용

### 머신러닝 모델링
[Kaggle Ref](https://flonelin.wordpress.com/2016/07/26/tuning-xgboostextream-gradient-boosting/)


```r
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
```

==================
## 2단계 캐글 소개 및 적용

### 머신러닝 모델링


```r
cv_scores = list()
train_df <- train[features_to_use]
train_y <- train['interest_level'] %>% mutate(interest_level = as.numeric(interest_level)-1)


kf <- crossv_kfold(train_df, 5) #modelr 사용!!!

kf[[1]][1]
```

```
$`1`
<resample [39,481 x 22]> 1, 2, 3, 5, 6, 7, 9, 10, 12, 14, ...
```

```r
kf[[2]][1]
```

```
$`1`
<resample [9,871 x 22]> 4, 8, 11, 13, 24, 25, 33, 34, 35, 40, ...
```


==================
## 2단계 캐글 소개 및 적용

### 머신러닝 모델링


```r
#caret 사용시
kf <- createFolds(1:dim(train_y)[1], k = 5)
str(kf)
```

```
List of 5
 $ Fold1: int [1:9871] 3 5 10 16 18 25 27 31 34 35 ...
 $ Fold2: int [1:9870] 1 6 7 15 23 52 55 66 72 83 ...
 $ Fold3: int [1:9870] 13 17 28 29 33 41 54 58 59 67 ...
 $ Fold4: int [1:9870] 4 8 9 32 36 37 38 40 43 47 ...
 $ Fold5: int [1:9871] 2 11 12 14 19 20 21 22 24 26 ...
```

```r
#modelr crossv_mc 사용시 
#Monte Carlo cross-validation
kf <- crossv_mc(train_df, 5)
kf
```

```
# A tibble: 5 x 3
           train           test   .id
          <list>         <list> <chr>
1 <S3: resample> <S3: resample>     1
2 <S3: resample> <S3: resample>     2
3 <S3: resample> <S3: resample>     3
4 <S3: resample> <S3: resample>     4
5 <S3: resample> <S3: resample>     5
```

```r
#modelr crossv_kfold 사용시 
kf <- crossv_kfold(train_df, 5)
kf
```

```
# A tibble: 5 x 3
           train           test   .id
          <list>         <list> <chr>
1 <S3: resample> <S3: resample>     1
2 <S3: resample> <S3: resample>     2
3 <S3: resample> <S3: resample>     3
4 <S3: resample> <S3: resample>     4
5 <S3: resample> <S3: resample>     5
```

- caret과 비교했을 때 cross validatoin 기능이 비슷하지만 차이가 있음

- caret는 train 모델에 컨트롤 옵션에서 cv를 할 수 있도록 권장하고 있음 (repeatedcv등)

- modelr은 column-list 방식을 제공하고 있어서 좀 더 범용적으로 사용하기 쉬움

==================
## 2단계 캐글 소개 및 적용

### 머신러닝 모델링


```r
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
```

```
[1]	train-mlogloss:1.076439	test-mlogloss:1.076825 
Multiple eval metrics are present. Will use test_mlogloss for early stopping.
Will train until test_mlogloss hasn't improved in 20 rounds.

[2]	train-mlogloss:1.055277	test-mlogloss:1.056040 
[3]	train-mlogloss:1.035207	test-mlogloss:1.036291 
[4]	train-mlogloss:1.016148	test-mlogloss:1.017593 
[5]	train-mlogloss:0.998039	test-mlogloss:0.999809 
[6]	train-mlogloss:0.980737	test-mlogloss:0.982852 
[7]	train-mlogloss:0.964136	test-mlogloss:0.966587 
[8]	train-mlogloss:0.948357	test-mlogloss:0.951158 
[9]	train-mlogloss:0.933270	test-mlogloss:0.936396 
[10]	train-mlogloss:0.918841	test-mlogloss:0.922260 
[11]	train-mlogloss:0.905106	test-mlogloss:0.908771 
[12]	train-mlogloss:0.891869	test-mlogloss:0.895844 
[13]	train-mlogloss:0.879216	test-mlogloss:0.883443 
[14]	train-mlogloss:0.867088	test-mlogloss:0.871562 
[15]	train-mlogloss:0.855374	test-mlogloss:0.860153 
[16]	train-mlogloss:0.844140	test-mlogloss:0.849205 
[17]	train-mlogloss:0.833379	test-mlogloss:0.838632 
[18]	train-mlogloss:0.823037	test-mlogloss:0.828506 
[19]	train-mlogloss:0.813177	test-mlogloss:0.818860 
[20]	train-mlogloss:0.803625	test-mlogloss:0.809552 
[21]	train-mlogloss:0.794431	test-mlogloss:0.800645 
[22]	train-mlogloss:0.785619	test-mlogloss:0.792100 
[23]	train-mlogloss:0.777107	test-mlogloss:0.783845 
[24]	train-mlogloss:0.768891	test-mlogloss:0.775936 
[25]	train-mlogloss:0.761038	test-mlogloss:0.768347 
[26]	train-mlogloss:0.753368	test-mlogloss:0.760978 
[27]	train-mlogloss:0.746082	test-mlogloss:0.753940 
[28]	train-mlogloss:0.739020	test-mlogloss:0.747067 
[29]	train-mlogloss:0.732187	test-mlogloss:0.740503 
[30]	train-mlogloss:0.725602	test-mlogloss:0.734151 
[31]	train-mlogloss:0.719262	test-mlogloss:0.727998 
[32]	train-mlogloss:0.713127	test-mlogloss:0.722098 
[33]	train-mlogloss:0.707217	test-mlogloss:0.716366 
[34]	train-mlogloss:0.701492	test-mlogloss:0.710848 
[35]	train-mlogloss:0.695990	test-mlogloss:0.705530 
[36]	train-mlogloss:0.690692	test-mlogloss:0.700460 
[37]	train-mlogloss:0.685583	test-mlogloss:0.695534 
[38]	train-mlogloss:0.680632	test-mlogloss:0.690803 
[39]	train-mlogloss:0.675845	test-mlogloss:0.686232 
[40]	train-mlogloss:0.671177	test-mlogloss:0.681744 
[41]	train-mlogloss:0.666732	test-mlogloss:0.677476 
[42]	train-mlogloss:0.662380	test-mlogloss:0.673354 
[43]	train-mlogloss:0.658197	test-mlogloss:0.669428 
[44]	train-mlogloss:0.654114	test-mlogloss:0.665580 
[45]	train-mlogloss:0.650177	test-mlogloss:0.661796 
[46]	train-mlogloss:0.646330	test-mlogloss:0.658128 
[47]	train-mlogloss:0.642672	test-mlogloss:0.654662 
[48]	train-mlogloss:0.639125	test-mlogloss:0.651271 
[49]	train-mlogloss:0.635643	test-mlogloss:0.647961 
[50]	train-mlogloss:0.632312	test-mlogloss:0.644857 
[51]	train-mlogloss:0.629088	test-mlogloss:0.641763 
[52]	train-mlogloss:0.625966	test-mlogloss:0.638788 
[53]	train-mlogloss:0.622886	test-mlogloss:0.635886 
[54]	train-mlogloss:0.619971	test-mlogloss:0.633125 
[55]	train-mlogloss:0.617075	test-mlogloss:0.630387 
[56]	train-mlogloss:0.614298	test-mlogloss:0.627783 
[57]	train-mlogloss:0.611606	test-mlogloss:0.625304 
[58]	train-mlogloss:0.608973	test-mlogloss:0.622837 
[59]	train-mlogloss:0.606437	test-mlogloss:0.620514 
[60]	train-mlogloss:0.603945	test-mlogloss:0.618196 
[61]	train-mlogloss:0.601530	test-mlogloss:0.615976 
[62]	train-mlogloss:0.599193	test-mlogloss:0.613813 
[63]	train-mlogloss:0.596934	test-mlogloss:0.611754 
[64]	train-mlogloss:0.594675	test-mlogloss:0.609708 
[65]	train-mlogloss:0.592538	test-mlogloss:0.607733 
[66]	train-mlogloss:0.590478	test-mlogloss:0.605824 
[67]	train-mlogloss:0.588496	test-mlogloss:0.603999 
[68]	train-mlogloss:0.586498	test-mlogloss:0.602161 
[69]	train-mlogloss:0.584589	test-mlogloss:0.600429 
[70]	train-mlogloss:0.582741	test-mlogloss:0.598748 
[71]	train-mlogloss:0.580961	test-mlogloss:0.597122 
[72]	train-mlogloss:0.579244	test-mlogloss:0.595535 
[73]	train-mlogloss:0.577576	test-mlogloss:0.594021 
[74]	train-mlogloss:0.575940	test-mlogloss:0.592570 
[75]	train-mlogloss:0.574363	test-mlogloss:0.591120 
[76]	train-mlogloss:0.572796	test-mlogloss:0.589713 
[77]	train-mlogloss:0.571271	test-mlogloss:0.588293 
[78]	train-mlogloss:0.569763	test-mlogloss:0.586901 
[79]	train-mlogloss:0.568344	test-mlogloss:0.585645 
[80]	train-mlogloss:0.566888	test-mlogloss:0.584372 
[81]	train-mlogloss:0.565516	test-mlogloss:0.583147 
[82]	train-mlogloss:0.564191	test-mlogloss:0.581967 
[83]	train-mlogloss:0.562896	test-mlogloss:0.580838 
[84]	train-mlogloss:0.561624	test-mlogloss:0.579762 
[85]	train-mlogloss:0.560387	test-mlogloss:0.578720 
[86]	train-mlogloss:0.559177	test-mlogloss:0.577647 
[87]	train-mlogloss:0.558018	test-mlogloss:0.576644 
[88]	train-mlogloss:0.556856	test-mlogloss:0.575660 
[89]	train-mlogloss:0.555785	test-mlogloss:0.574735 
[90]	train-mlogloss:0.554695	test-mlogloss:0.573770 
[91]	train-mlogloss:0.553578	test-mlogloss:0.572846 
[92]	train-mlogloss:0.552566	test-mlogloss:0.571930 
[93]	train-mlogloss:0.551535	test-mlogloss:0.571027 
[94]	train-mlogloss:0.550526	test-mlogloss:0.570172 
[95]	train-mlogloss:0.549461	test-mlogloss:0.569267 
[96]	train-mlogloss:0.548502	test-mlogloss:0.568418 
[97]	train-mlogloss:0.547559	test-mlogloss:0.567612 
[98]	train-mlogloss:0.546597	test-mlogloss:0.566801 
[99]	train-mlogloss:0.545750	test-mlogloss:0.566066 
[100]	train-mlogloss:0.544896	test-mlogloss:0.565397 
[[1]]
[1] 0.5653974
```

==================
## 2단계 캐글 소개 및 적용

### 대회 제출


```r
test_df <- test[features_to_use]

result_test = runXGB(as.matrix(train_df), as.matrix(train_y), as.matrix(test_df), num_rounds=100)
out_df = as.data.frame(matrix(result_test[[1]], ncol=3, byrow=T))
colnames(out_df) = c("low", "medium", "high")
out_df['listing_id'] <- test_df %>% select(listing_id)
write.csv(out_df, file='data/xgb_starter_num_rounds100_xwmooc_1meetup.csv', row.names = FALSE)
```

![제출 결과](./img/img2.png)


==================

# The End
## modelr을 캐글에 적용하기

- 이상열 : syleeie@gmail.com

- 캐글즐기기 매주 수요일 (파트4 종료, https://github.com/KaggleBreak/analyticstool)

- 텐서뽀개기 격주 화요일 (6/27일 파트1 시작,  https://github.com/KaggleBreak/tensorbreak)

[이벤트 링크](https://www.facebook.com/events/1945163935730064)

- 워킹캐글 주말 (8/26일 파트2 시작, https://github.com/KaggleBreak/walkingkaggle)

[이벤트 링크](https://www.facebook.com/events/1472119989567282)

- Q&A?

==================
## Appendix

<br>

### R4DS 24, 25장 요약
- model_building
- many_models
- 회귀 모형 위주로 column-list 여러가지 방법에 대해 자세히 나와 있음. 함수형 프로그램, 분석 자동화

[R4DS](https://github.com/KaggleBreak/analyticstool/blob/master/part4/R/analytics/20170719/23_24.model_building_many_models.nb.html)

- [broom 소개](https://cran.r-project.org/web/packages/broom/vignettes/broom.html)

- Convert statistical analysis objects from R into tidy data frames, so that they can more easily be combined, reshaped and otherwise processed with tools like 'dplyr', 'tidyr' and 'ggplot2'
