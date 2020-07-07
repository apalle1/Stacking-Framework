# Load in our libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# Going to use these 5 models for first-Level Predictions
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
# Going to use Xgboost for second-Level Predictions
import xgboost as xgb

# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
TARGET_COL_NAME = 'target'

############################### Helpers via Python Classes ###############################
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

############################### Out-of-Fold Predictions ###############################
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain, 1))
    oof_test = np.zeros((ntest, 1))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for fold_, (trn_, val_) in enumerate(kf.split(X=ntrain)):
        print("Current Fold: {}".format(fold_))
        trn_x, trn_y = x_train[trn_, :], y_train[trn_]
        val_x, val_y = x_train[val_, :], y_train[val_]

        # FIT MODEL HERE
        clf.train(trn_x, trn_y)

        # GENERATE PREDICTIONS FOR VALIDATION DATA
        oof_train[val_] = clf.predict(val_x)
        # GENERATE PREDICTIONS FOR TEST DATA
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


############################### Generating First-Level Models ###############################
# Define parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025}

# Create 5 objects
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Creating NumPy arrays out of our train and test sets
y_train = train[TARGET_COL_NAME].ravel()
train = train.drop([TARGET_COL_NAME], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Output of the First level Predictions
# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

############################### Second-Level Predictions from the First-level Output ###############################
x_train = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(#learning_rate = 0.02,
                        n_estimators= 2000,
                        max_depth= 4,
                        min_child_weight= 2,
                        #gamma=1,
                        gamma=0.9,                        
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective= 'binary:logistic',
                        nthread= -1,
                        scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)