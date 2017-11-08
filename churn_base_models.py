import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.metrics import roc_auc_score as auc
import gc
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.preprocessing import StandardScaler as SS
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
sample_sub = pd.read_csv('../input/sample_submission_fBo3EW5.csv')

Y= df_train.Responders.values
tr_ids =df_train.UCIC_ID.values
te_ids=df_test.UCIC_ID.values
df_train.drop(['Responders', 'UCIC_ID'], axis=1, inplace=True)
df_test.drop(['UCIC_ID'], axis=1, inplace=True)


to_drop=[]
for c in df_train.columns:
    if (df_train[c].nunique() == 0) | (df_test[c].nunique()==0):
        to_drop.append(c)
    elif (df_train[c].nunique() == 1) & (df_train[c].dtype == 'O'): 
        df_train[c].replace({'Y':1}, inplace=True)
        df_test[c].replace({'Y':1}, inplace=True)

df_train.drop(to_drop, axis=1, inplace=True)
df_test.drop(to_drop, axis=1, inplace=True)


df_full = pd.concat((df_train, df_test)).reset_index(drop=True)

to_drop=[]
for c in df_full.columns:
    if (float(df_full[c].isnull().sum())/df_full.shape[0]) >=0.99:
        to_drop.append(c)
print(to_drop)

df_full.drop(to_drop, axis=1,inplace=True)


for c in list(df_full.select_dtypes(include=['object']).columns):
    df_full[c] = pd.factorize(df_full[c].values, sort=True)[0]


print(df_full.shape)


df_full = df_full.fillna(-999) #-999

df_full['CR_AMB_Drop_EOP_1'] = df_full.CR_AMB_Drop_Build_1 * df_full.EOP_prev1
df_full['I_CR_AQB_EOP_1'] = df_full.I_CR_AQB_PrevQ1 / df_full.EOP_prev1
df_full['I_AQB_EOP_1'] = df_full.I_AQB_PrevQ1 / df_full.EOP_prev1
df_full['D_Prev1_EOP_1'] = df_full.D_prev1 / df_full.EOP_prev1
df_full['CR_AMB_Drop_1_D_prev1'] = df_full.CR_AMB_Drop_Build_1 / df_full.D_prev1
df_full['CR_AMB_Drop_2_D_prev1'] = df_full.CR_AMB_Drop_Build_2 / df_full.D_prev1
df_full['CR_AMB_Drop_1_vintage'] = df_full.CR_AMB_Drop_Build_1 / df_full.vintage


df_full = df_full.replace([np.inf, -np.inf], np.nan)
df_full = df_full.fillna(-999)
df_full = SS().fit_transform(df_full)

df_train = df_full[:300000]
df_test = df_full[300000:]

gc.collect()


lgb_train = lgb.Dataset(df_train, Y)

lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'binary',
    'nthread': -1, 'silent': True,
    'num_leaves': 2**8 -1, 'learning_rate': 0.02, 'max_depth': 8,
    'max_bin': 2**8 -1, 'metric': 'auc',
    'colsample_bytree': 0.33, #0.4
    'bagging_fraction': 0.9, 
    'bagging_freq': 10, 
    'scale_pos_weight': 1.02, 
    'bagging_seed': 619, #619
    'feature_fraction_seed': 619 #619
    }
    
nrounds = 2000  
kfolds = 5  
oof_train=pd.DataFrame({'UCIC_ID': tr_ids, 'Responders':0})
best=[]
score=[]

skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train[train_index], df_train[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    ltrain = lgb.Dataset(X_train,y_train)
    lval = lgb.Dataset(X_val,y_val, reference= ltrain)

    gbdt = lgb.train(lgb_params, ltrain, nrounds, valid_sets=lval,
                         verbose_eval=100,
                         early_stopping_rounds=30)  
    bst=gbdt.best_iteration
    pred=gbdt.predict(X_val, num_iteration=bst)
    oof_train.loc[test_index,"Responders"]= pred
    
    scr=auc(y_val,pred) 
    
    best.append(bst)    
    score.append(scr)
    i+=1
    
    del ltrain
    del lval
    del gbdt
    gc.collect()

print(np.mean(score))
print(np.mean(best))

oof_train.to_csv('../output/lgb_deep_oof_v1.csv', index=False)


best_nrounds=int(round(np.mean(best)))

gbdt = lgb.train(lgb_params, lgb_train, best_nrounds,verbose_eval=50)
pred=gbdt.predict(df_test)
submit=pd.DataFrame({'UCIC_ID': te_ids, 'Responders':pred})

submit.to_csv('../output/lgb_deep_test_v1.csv', index=False)


lgb_params2 = {
    'boosting_type': 'gbdt', 'objective': 'binary',
    'nthread': -1, 'silent': True,
    'num_leaves': 2**6 -1, 'learning_rate': 0.03, 'max_depth': 5,
    'max_bin': 2**6 -1, 'metric': 'auc',
    'colsample_bytree': 0.6, #0.4
    'bagging_fraction': 0.85, 
    'bagging_freq': 10, 
    'scale_pos_weight': 1.02, 
    'bagging_seed': 1729, 
    'feature_fraction_seed': 1729 
    }
    
nrounds = 2000  
kfolds = 5  
oof_train=pd.DataFrame({'UCIC_ID': tr_ids, 'Responders':0})
best=[]
score=[]


skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train[train_index], df_train[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    ltrain = lgb.Dataset(X_train,y_train)
    lval = lgb.Dataset(X_val,y_val, reference= ltrain)

    gbdt = lgb.train(lgb_params2, ltrain, nrounds, valid_sets=lval,
                         verbose_eval=100,
                         early_stopping_rounds=30)  
    bst=gbdt.best_iteration
    pred=gbdt.predict(X_val, num_iteration=bst)
    oof_train.loc[test_index,"Responders"]= pred
    
    scr=auc(y_val,pred) 
    
    best.append(bst)    
    score.append(scr)
    i+=1
    
    del ltrain
    del lval
    del gbdt
    gc.collect()

print(np.mean(score))
print(np.mean(best))

oof_train.to_csv('../output/lgb_shallow_oof_v1.csv', index=False)


best_nrounds=int(round(np.mean(best)))

gbdt = lgb.train(lgb_params2, lgb_train, best_nrounds,verbose_eval=50)
pred=gbdt.predict(df_test)
submit=pd.DataFrame({'UCIC_ID': te_ids, 'Responders':pred})

submit.to_csv('../output/lgb_shallow_test_v1.csv', index=False)


X_train=xgb.DMatrix(df_train,Y)
X_test=xgb.DMatrix(df_test)

xgb_params = {
    'tree_method':'hist',
    'seed': 2580, 
    'colsample_bytree': 0.5,
    'silent': 1,
    'subsample': 0.8,
    'learning_rate': 0.03, 
    'objective': 'binary:logistic',
    'max_depth': 6,
    'min_child_weight': 5, 
    'gamma': 0.02, 
    'alpha': 0.02,
    'eval_metric' : 'auc'
    
}

oof_train=pd.DataFrame({'UCIC_ID': tr_ids, 'Responders':0})
best=[]
score=[]


skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train[train_index], df_train[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    dtrain = xgb.DMatrix(X_train,y_train)
    dval = xgb.DMatrix(X_val,y_val)
    watchlist = [(dtrain, 'train'), (dval, 'eval')]

    gbdt = xgb.train(xgb_params, dtrain, nrounds, watchlist,
                         verbose_eval=100,
                         early_stopping_rounds=30)  
    bst=gbdt.best_ntree_limit
    pred=gbdt.predict(dval, ntree_limit=bst)
    oof_train.loc[test_index,"Responders"]= pred
    
    scr=auc(y_val,pred) 
    
    best.append(bst)    
    score.append(scr)
    i+=1
    
    del dtrain
    del dval
    del gbdt
    gc.collect()
    
print(np.mean(score))
print(np.mean(best))

oof_train.to_csv('../output/xgb_medium_oof_v1.csv', index=False)


best_nrounds=int(round(np.mean(best)))
X_train=xgb.DMatrix(df_train,Y)

gbdt = xgb.train(xgb_params, X_train, best_nrounds,verbose_eval=50)
pred=gbdt.predict(X_test)
submit=pd.DataFrame({'UCIC_ID': te_ids, 'Responders':pred})

submit.to_csv('../output/xgb_medium_test_v1.csv', index=False)


RF=RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=3, min_samples_leaf=2,
                         max_features='auto', oob_score=True, n_jobs=-1, random_state=3492, verbose=1)                        

oof_train=pd.DataFrame({'UCIC_ID': tr_ids, 'Responders':0})
score=[]
skf = SKF( n_splits=kfolds, shuffle=True,random_state=123)
i=0
for train_index, test_index in skf.split(df_train, Y):
    print('Fold {0}'.format(i + 1))
    X_train, X_val = df_train[train_index], df_train[test_index]
    y_train, y_val = Y[train_index],Y[test_index]

    RF.fit(X_train, y_train)
    pred=RF.predict_proba(X_val)
    oof_train.loc[test_index,"Responders"]= pred[:,1]

    scr=auc(y_val,pred[:,1])         
    score.append(scr)
    i+=1
    
    gc.collect()

print(np.mean(score))


oof_train.to_csv('../output/RF_oof_v1.csv', index=False)

RF.fit(df_train,Y)
pred=RF.predict_proba(df_test)[:,1]
submit=pd.DataFrame({'UCIC_ID': te_ids, 'Responders':pred})

submit.to_csv('../output/RF_test_v1.csv', index=False)


