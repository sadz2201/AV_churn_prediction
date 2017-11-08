import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
import gc
import warnings
warnings.filterwarnings("ignore")


actual = pd.read_csv('../input/train.csv')[['UCIC_ID','Responders']]
y=actual.Responders.values

lgb1 = pd.read_csv('../output/lgb_deep_oof_v1.csv') 
lgb1_t = pd.read_csv('../output/lgb_deep_test_v1.csv') 

lgb2 = pd.read_csv('../output/lgb_shallow_oof_v1.csv') 
lgb2_t = pd.read_csv('../output/lgb_shallow_test_v1.csv') 

xgb = pd.read_csv('../output/xgb_medium_oof_v1.csv') 
xgb_t = pd.read_csv('../output/xgb_medium_test_v1.csv') 

RF = pd.read_csv('../output/RF_oof_v1.csv')  
RF_t = pd.read_csv('../output/RF_test_v1.csv')


S_train = np.column_stack((lgb1.Responders.values,lgb2.Responders.values,xgb.Responders.values, RF.Responders.values))
print(S_train.shape)

S_test = np.column_stack((lgb1_t.Responders.values,lgb2_t.Responders.values,xgb_t.Responders.values, RF_t.Responders.values))
print(S_test.shape)

print(auc(y, lgb1.Responders))
print(auc(y, lgb2.Responders))
print(auc(y, xgb.Responders))
print(auc(y, RF.Responders))

def scorer(labels, probs):
    decile_cut = np.percentile(probs, 80)
    decile_labels = y[probs > decile_cut]
    return float(sum(decile_labels)) / sum(labels)

print(scorer(y, lgb1.Responders))
print(scorer(y, lgb2.Responders))
print(scorer(y, xgb.Responders))
print(scorer(y, RF.Responders))

S_train = np.log(S_train/(1-S_train))
S_test = np.log(S_test/(1-S_test))

def make_scorer(estimator, x, y):
    probs = estimator.predict_proba(x)[:, 1]
    decile_cut = np.percentile(probs, 80)
    decile_labels = y[probs > decile_cut]
    return float(sum(decile_labels)) / sum(y)

results_auc = cross_val_score(LogisticRegression(random_state = 2017),S_train, y, cv=5, scoring='roc_auc')
results_lift = cross_val_score(LogisticRegression(random_state = 2017), S_train, y, cv=5, scoring=make_scorer)
print("Stacker AUC score: %.5f" % (results_auc.mean()))
print("Stacker Lift score: %.5f" % (results_lift.mean()))

stacker=LogisticRegression(random_state = 2017)
stacker.fit(S_train, y)
res = stacker.predict_proba(S_test)[:, 1]

print(stacker.coef_)
print(stacker.intercept_)

submit = pd.DataFrame({'UCIC_ID': lgb1_t.UCIC_ID, 'Responders': res})
submit.to_csv('../output/stacking_2lgb_1xgb_1rf_final.csv', index=False)


