import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sub_ids = test['portfolio_id']
test = test.drop('portfolio_id', axis=1)
y = train['return']
train = train.drop(['portfolio_id','return'], axis=1)

# remove some  fields  for simplicity
train = train.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)
test = test.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)

# handle missing values 
train['hedge_value'].fillna(False, inplace=True)
test['hedge_value'].fillna(False, inplace=True)

# missing values for numeric fields
train['sold'].fillna(train['sold'].median(), inplace=True)
train['bought'].fillna(train['bought'].median(), inplace=True)
train['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)
test['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)

# encode categorical fields
obj_cols = [x for x in train.columns if train[x].dtype == 'object']
encoder = LabelEncoder()
for x in obj_cols:
    encoder.fit(train[x])
    train[x] = encoder.transform(train[x])
    test[x] = encoder.transform(test[x])


#-------------------------Random Forest Code-------------------------------------
#--------------------------------------------------------------------------------
#Random Forest below 4 lines
forest_reg = RandomForestRegressor(random_state=7)
scores = cross_val_score(forest_reg, train, y, scoring='r2', cv=5)
print(scores)
print('mean r2:',np.mean(scores))

#below 3 lines are for Random Forest
forest_reg = RandomForestRegressor(random_state=7)
forest_reg.fit(train, y)
preds = forest_reg.predict(test)
#Random Forest
sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': preds})
#--------------------------------------------------------------------------------
#------------------------Random Forest Code Finish-------------------------------



#---------------------------XGB Regressor-----------------------------------------
#Do not comment this line (XGB Regressor)
#regr = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0,learning_rate=0.05, max_depth=6,min_child_weight=1.5,                       n_estimators=7200,                       reg_alpha=0.9,     reg_lambda=0.6,                       subsample=0.2,seed=42,                       silent=1)
#regr.fit(train, y)
# run prediction on training set to get an idea of how well it does (XGB Regressor)
#y_pred = regr.predict(test)
#sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': y_pred})

#----------------------------XGB Regressor Code Finish-----------------------------	

#----------------------------Lasso Regression-------------------------------------
#best_alpha = 0.0099

#regr = Lasso(alpha=best_alpha, max_iter=500000)
#regr.fit(train, y)
#y_pred = regr.predict(test)
#lasso_ex = np.exp(y_pred)
#sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': y_pred})

#------------------------------------Neural Network-------------------------------
#---------------------------------------------------------------------------------
#np.random.seed(10)

#create Model
#define base model
#def base_model():
#     model = Sequential()
#     model.add(Dense(20, input_dim=398, init='normal', activation='relu'))
#     model.add(Dense(10, init='normal', activation='relu'))
#     model.add(Dense(1, init='normal'))
#     model.compile(loss='mean_squared_error', optimizer = 'adam')
#     return model

#seed = 7
#np.random.seed(seed)

#scale = StandardScaler()

#X_train = scale.fit_transform(train)
#X_test = scale.fit_transform(test)

#keras_label = y.as_matrix()
#clf = KerasRegressor(build_fn=base_model, nb_epoch=1000, batch_size=5,verbose=0)
#clf.fit(X_train,keras_label)

#make predictions and create the submission file 
#kpred = clf.predict(X_test) 
#kpred = np.exp(kpred)
#sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': kpred})
#----------------------------------Neural Network--------------------------
#To save a file as csv
filename = 'sub_returns.csv'
sub.to_csv(filename, index=False)
