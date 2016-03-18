library(caret)
library(corrplot)
library(ggplot2)
library(reshape)


setwd("C:/Users/vinay.benny/Documents/Kaggle/Santader")

train <- read.csv("train.csv");
test <- read.csv("test.csv");

cat("removing constant features\n")
toRemove <- c()
feature.names <- names(train)
for (f in feature.names) {
  if (sd(train[[f]])==0) {
    toRemove <- c(toRemove,f)
    cat(f,"is constant\n")
  }
}

train.names <- setdiff(names(train),toRemove)
test.names <- setdiff(names(test),toRemove)
train <- train[, train.names]
test <- test[, test.names]

correl <- cor(train)
correl.m <- melt(correl)
ggplot(correl.m, aes(X1, X2, fill = value)) + geom_tile() + 
  scale_fill_gradient2(low = "red",  high = "blue")

correlattr <- findCorrelation(correl, cutoff = .999, verbose = FALSE);