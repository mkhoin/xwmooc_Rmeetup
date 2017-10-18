
library(tm)
library(tidyverse)
library(DT) #data table
library(stringr) # string 처리
library(tidytext)
#library(ggthemes)
#library(extrafont)
#loadfonts()
#install.packages('tidytext')
#install.packages('extrafont')

getwd()

library(dplyr)
library(tidyr)
library(purrr)
library(readr)

Ted_edu_eng <- "/Users/hansumi/Downloads/Ted_edu_eng/"

# Define a function to read all files from a folder into a data frame
read_folder <- function(infolder){
    data_frame(file = dir(infolder, full.names = TRUE)) %>%
    mutate(text = map(file, read_lines)) %>%
    transmute(id = basename(file), text) %>%
    unnest(text)
}

# unnest() and map() to apply read_folder to each subfolder

raw_text <- data_frame(folder= dir(Ted_edu_eng, full.names = TRUE)) %>%
    unnest(map(folder, read_folder)) %>%
    transmute(category = basename(folder), id, text)    #category: Science, Non_Science
raw_text

library(ggplot2)

raw_text %>%
  group_by(category) %>%
  summarize(total_files = n_distinct(id))  %>%
  ggplot(aes(category, total_files)) +
  geom_col() +
  coord_flip()
#total_files: Science: 340; Non_Science: 248

# unnest_tokens() for tokenization to split the dataset into tokens while removing stop words

library(tidytext)
data(stop_words)   

ted_words <- raw_text %>%
   unnest_tokens(word, text) %>%
   filter(str_detect(word, "[a-z']$"),
         !word %in% stop_words$word)   #stop_words 제거

ted_words

# Words in Science vs. Non-Science of Ted Edu Corpus
# Most Common Words for the whole corpus
ted_words %>%
 count(word, sort = TRUE)

# Words by Category
words_by_category <- ted_words %>%
 count(category, word, sort = TRUE) %>%
 ungroup()

words_by_category

#install.packages('wordcloud2')
#library(wordcloud2)

ted_word_count <- ted_words %>%
  count(word, sort = TRUE) %>%
  mutate(word = reorder(word, n)) 

#ted_word_count


library(wordcloud2)

# ted_words %>% 
#  anti_join(stop_words) %>%
#  count(word) %>%
#  with(wordcloud, n, max.words = 100)


ted_word_count %>%
 head(10) %>%
 wordcloud2() 

# Category differs in terms of topic/content
# tf-idf metric: the freq of words among categories

tf_idf <- words_by_category %>%
 bind_tf_idf(word, category, n) %>%
 arrange(desc(tf_idf))

tf_idf

#AFINN LEXICON_감성사전에 따라서 다른 결과값
category_sentiments <- words_by_category %>%
 inner_join(get_sentiments("afinn"), by = "word") %>%
 group_by(category) %>%
 summarize(score = sum(score * n)/sum(n))

# #BING LEXICON
# category_sentiments <- words_by_category %>%
#  inner_join(get_sentiments("bing"), by = "word") %>%
#  group_by(category) %>%
#  summarize(score = sum(score * n)/sum(n))
 
category_sentiments %>%
 mutate(category = reorder(category, score)) %>%
 ggplot(aes(category, score, fill = score >0)) +
 geom_col(show.legend = FALSE) +
 coord_flip() +
 ylab("Average sentiment score")

#get_sentiments("bing")

cleaned_text <- raw_text %>%
 group_by(category, id) %>%
 mutate(linenumber = row_number()) %>%
 ungroup()


# Tibble data가 되어야 하는 듯

ted_bigrams <- cleaned_text %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

ted_bigram_counts <- ted_bigrams %>%
 count(category, bigram, sort = TRUE) %>%
 ungroup() %>% 
 separate(bigram, c("word1", "word2"), sep = " ")  

#install.packages('wordcloud')
library(wordcloud)

raw_text %>%
 anti_join(stop_words) %>%
 count(word, by ="") %>%
 with(wordcloud(word, n, max.words = 100))
