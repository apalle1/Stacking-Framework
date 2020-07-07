# Stacked Generalization (Stacking)

Stacking uses first-level models to make predictions and then use these first-level predictions as features for 2nd level model(s). 
It helps you to combine different predictions of Y from first-level models as "metafeatures". To obtain metafeatures for train set, 
you split your data into K folds, and train on different K-1 parts each time while making prediction for 1 part that was left aside. 
To obtain metafeature for test set, you can average predictions from K test predictions (Variant A) or make single prediction based 
on all train data (Variant B). After that you train metaregressors on metafeatures and average predictions if you have several metaregressors.
Only the creation of the test set predictions is different between Variant A & B. That is, in both cases the creation of train set 
predictions stays the same.

* **Variant A**: In each fold we predict test set, so after completion of all folds we need to find mean (mode) of all temporary test set
predictions made in each fold.
* **Variant B**: We do not predict test set during cross-validation cycle. After completion of all folds we perform additional step: fit model
on full train set and predict test set once. This approach takes more time because we need to perform one additional fitting.

![alt text](https://github.com/apalle1/Stacking-Framework/blob/master/Variant%20A.PNG)

![alt text](https://github.com/apalle1/Stacking-Framework/blob/master/Variant%20B.PNG)

# References

https://mlwave.com/kaggle-ensembling-guide/

https://github.com/kaz-Anova/StackNet

https://www.kaggle.com/mmueller/stacking-starter

