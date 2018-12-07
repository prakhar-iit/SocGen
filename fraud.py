import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

sub_ids = test['transaction_id']
y = train['target']

train = train.drop(['transaction_id', 'target'], axis=1)
test = test.drop(['transaction_id'], axis=1)

y.value_counts(normalize=True)

cat_vars = [x for x in train.columns if 'cat_' in x]
len(cat_vars)

cat_to_drop_train = []
for x in cat_vars:
    if train[x].nunique() == 1:
        print(x, train[x].nunique())
        cat_to_drop_train.append(x)

cat_to_drop_test = []
for x in cat_vars:
    if test[x].nunique() == 1:
        print(x, test[x].nunique())
        cat_to_drop_test.append(x)

cat_to_drop = list(set(cat_to_drop_train + cat_to_drop_test))
train = train.drop(cat_to_drop, axis=1)
test = test.drop(cat_to_drop, axis=1)

print(train.shape)
print(test.shape)

cat_vars = [x for x in train.columns if 'cat_' in x]
len(cat_vars)

for x in cat_vars:
    train[x] = train[x].fillna('NaN')
    test[x] = test[x].fillna('NaN')
    encoder = LabelEncoder()
    encoder.fit(list(set(list(train[x]) + list(test[x]))))
    train[x] = encoder.transform(train[x])
    test[x] = encoder.transform(test[x])

forest_clf = RandomForestClassifier(random_state=7)

y_probas_forest = cross_val_predict(forest_clf, train, y, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]

roc_auc_score(y, y_scores_forest)

forest_clf = RandomForestClassifier(random_state=7)
forest_clf.fit(train, y)

preds = forest_clf.predict_proba(test)[:,1]

sub = pd.DataFrame({'transaction_id': sub_ids, 'target': preds})
sub = sub[['transaction_id','target']]    

filename='sub1.csv'
sub.to_csv(filename, index=False)
