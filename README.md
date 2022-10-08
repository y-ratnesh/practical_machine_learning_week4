# practical_machine_learning_week4

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


