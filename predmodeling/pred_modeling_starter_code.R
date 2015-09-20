#train <- read.csv2("http://freakonometrics.free.fr/training.csv",stringsAsFactors = FALSE)
train <- read.csv2("~/R/CRUG/insurance/training.csv",stringsAsFactors = FALSE)

# http://f.hypotheses.org/wp-content/blogs.dir/253/files/2015/08/Pricing_ENG.pdf
library(dplyr)

head(train)

# count distinct policy numbers
dtrain <- train %>% 
  distinct(PolNum) %>% 
  count()

print(paste("dup count: ",nrow(train) - dtrain))

# get dups
# get the dup records
# observe that Indppd is dissimilar field, assuming max() is correct
dtrain_dups <- train %>% 
  group_by(PolNum) %>%
  summarise(n=n()) %>%
  subset(n>1) %>% 
  ungroup() %>% 
  left_join(train,by="PolNum") %>%
  arrange(PolNum)

dtrain_dups <- dtrain_dups %>%
  group_by(PolNum, n, CalYear,Gender,Type,Category,Occupation,
           Age, Group1, Bonus, Poldur, Value, Adind, SubGroup2, 
           Group2, Density, Exppdays, Numtppd, Numtpbi, Indtpbi) %>%
  summarise(Indtppd = max(Indtppd))
             

claims <- train %>% 
  group_by(PolNum) %>%
  summarise(n=n()) %>%
  subset(n==1) %>% # removes dups
  ungroup() %>%
  left_join(train,by="PolNum") %>%
  union(dtrain_dups) %>%  # adds the corrected dups back in
  select(-n)


# convert character variables to numeric
claims$Indtpbi <- as.numeric(claims$Indtpbi)
claims$Density <- as.numeric(claims$Density)
claims$Indtppd <- as.numeric(claims$Indtppd)

hist(log(claims$Indtpbi))
hist(log(claims$Indtppd))

# create dependent variable
claims$claim <- ifelse(claims$Numtppd > 0,"Bad",
                      ifelse(claims$Numtpbi>0,"Bad","Good"))

summary(claims)
claims$claim <- as.factor(claims$claim )

# roughly follow this blog post to build a logit model:
# http://datascienceplus.com/perform-logistic-regression-in-r/
logit_model <- glm(claim ~.,
                   family=binomial(link='logit'),
                   data=subset(claims, select=c( -PolNum,-Indtppd,-Indtpbi,-Numtppd,-Numtpbi,-SubGroup2 ) ))

anova(logit_model, test="Chisq")

library(ROCR)
p <- predict(logit_model, 
             newdata=subset(claims, select=c( -PolNum,-Indtppd,-Indtpbi,-Numtppd,-Numtpbi,-SubGroup2 )),
            type="response")
pr <- prediction(p, claims$claim)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

# a little more clean up for caret 
claims_train <- subset(claims, select=c( -PolNum,-Indtppd,-Indtpbi,-Numtppd,-Numtpbi,-SubGroup2 ))
claims_train$claim <- as.factor(claims_train$claim)

# Stochastic Gradient Boosting
library(caret)
library(doMC)
registerDoMC(cores = 8)

fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5)


#tuningParameters <- expand.grid(interaction.depth = c(1,3,5),
#                                n.trees = c(100,300,500,1000),
#                                shrinkage = c(0.01, 0.1), 
#                                n.minobsinnode = 10)

tuningParameters <- expand.grid(interaction.depth = c(1,3,5),
                                n.trees = c(100,200,400,600,800,1000,1200,1500,2000),
                                shrinkage = c(0.01, 0.02,0.04,0.05, 0.08, 0.1, 0.12, 0.15), 
                                n.minobsinnode = 10)

gbmFit1 <- train(claim ~., 
                 data = claims_train,
                 method = "gbm",
                 trControl = fitControl,
                 tuneGrid = tuningParameters,
                 verbose = FALSE)
gbmFit1

trellis.par.set(caretTheme())
plot(gbmFit1)


# recreate logit curve
p <- predict(logit_model, 
             newdata=subset(claims, select=c( -PolNum,-Indtppd,-Indtpbi,-Numtppd,-Numtpbi,-SubGroup2 )),
             type="response")
pr <- prediction(p, claims$claim)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")

# create ROC curve with GBM model
p2 <- predict(gbmFit1, 
             newdata=claims_train,
             type="prob")
pr2 <- prediction(p2[-1], claims_train$claim)
prf2 <- performance(pr2, measure = "tpr", x.measure = "fpr")

plot(prf)
plot(prf2, add = TRUE)

auc2 <- performance(pr2, measure = "auc")
auc2 <- auc2@y.values[[1]]
auc2
auc

attributes(gbmFit1)
gbmFit1$xlevels

