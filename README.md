# practical_machine_learning_week4
Practical Machine Learning
================
Ratnesh
08-10-2022



## Abstract

In this analysis we try to predict the manner in which people exercise
based on accelerometers on the belt, forearm, arm, and dumbell of 6
participants from the Weight Lifting Exercise Dataset using different
machine learning algorithms.

Six participants were asked to perform barbell lifts correcty and
incorrectly in five different manners wearing fitness trackers like
Jawbone Up, Nike FuelBand, and Fitbit in this dataset. The data gained
from this devices is used to train the models.

*If you want to run this analysis from this Markdown document please
make sure to have pml-testing.csv and pml-training.csv in the root
directory of this document. Furthermore the models trained by this
notebook will be saved in the root directory.*



## Conclusion

The overall performance of `rf` is superior compared to the other two
models, `rpart` and `glmnet`, on the training and validation dataset.
Overall the accuracy of `rpart` and `glmnet` is lower than I expected.
Performance of the `glmnet` could be further improved by choosing higher
`lambda`-values. Lower `lambda`-values provoke a warning message that no
convergence can be found after the maximum amount of iterations. Both
models could improve by choosing a better parameter set.

The performance of all models is is a bit lower on the validation data,
but overall near the accuracy of the training data implying a low OOS,
which is less than 1 % according to the model metrics for `rf` (0.08%,
see: Metrics for Random Forest), and rejecting an overfitting of the
train data split as seen in the validation data prediction accuracy.
This low OOS-error is as expected. Though I suspect, given extremly high
accuracy of the `rf` model, that this model would perform poorly in a
real world setting with other participiants and other measuring devices.
A problem known for machine learning. This is further described in [this
paper](https://arxiv.org/abs/2011.03395) for example.

This could be avoided by removing predictors that are highly specific to
this data set, like the `user_name`, `num_window` or `cvtd_timestamp`
and aquiring a much bigger data set with more devices and more diverse
users. `cvtd_timestamp` seems to have a reasonable high impact on the
prediction but has no impact in a real world setting (see: Predictor
Impact in the Annex).

Given the near perfect accuracy of the `rf` model on the validation
dataset no further modeling techniques like ensemble models or further
parameter tuning is performed and this model is choosen for predicting
the testing data set.

## Prediction on the testing data

``` r
tibble(N = 1:nrow(test), 
       Classe=predict(rfFit, test))
```

    ## # A tibble: 20 x 2
    ##        N Classe
    ##    <int> <fct> 
    ##  1     1 B     
    ##  2     2 A     
    ##  3     3 B     
    ##  4     4 A     
    ##  5     5 A     
    ##  6     6 E     
    ##  7     7 D     
    ##  8     8 B     
    ##  9     9 A     
    ## 10    10 A     
    ## 11    11 B     
    ## 12    12 C     
    ## 13    13 B     
    ## 14    14 A     
    ## 15    15 E     
    ## 16    16 E     
    ## 17    17 A     
    ## 18    18 B     
    ## 19    19 B     
    ## 20    20 B

## Annex

### Exploratory Plot

``` r
# Detect missing value fields

nClasses <- pml_training %>% count(classe)

par(mfrow=c(2,2))
barplot(missingValues, ylab="N(Only Missing Values)", xlab="Colname", main="Missing Values")
boxplot(sapply(pml_training %>% 
                 select_if(is.numeric), 
               function(x) (x - sd(x)) / (mean(x))),
        ylab="Normalized-Centered Values", xlab="Colname", main="Values")
barplot(n ~ classe, data=nClasses, ylab="N(Classes)", xlab="Class", main="Class Count")
```

![](Prediction-Assignment-Writeup_files/figure-gfm/Exploratory-1.png)<!-- -->

### Predictor Impact

``` r
samples <- c(826, 1870, 2215, 3068, 3759)
explainer <- lime(train, rfFit)
explanation <- explain(validate[c(samples[1]),], explainer, n_labels = 1, n_features = 10)
plot_features(explanation) +
  labs(title="Impact of Predictors",
       subtitle="Based on an example in the validation data set",
       caption="Index: 826")
```

![](Prediction-Assignment-Writeup_files/figure-gfm/PredictorImpact-1.png)<!-- -->

### Metrics for Random Forest

``` r
rfFit$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 24
    ## 
    ##         OOB estimate of  error rate: 0.08%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4464    0    0    0    0 0.0000000000
    ## B    2 3035    1    0    0 0.0009874918
    ## C    0    2 2735    1    0 0.0010956903
    ## D    0    0    2 2569    2 0.0015546055
    ## E    0    0    0    3 2883 0.0010395010
