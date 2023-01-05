import pandas as pd
import xgboost as xgb
from sklearn import preprocessing


train = pd.read_csv(r'D:\BaiduNetdiskDownload\CBD_Location\train.csv')
tests = pd.read_csv(r'D:\BaiduNetdiskDownload\CBD_Location\test.csv')

train['time_stamp'] = pd.to_datetime(pd.Series(train['time_stamp']))
tests['time_stamp'] = pd.to_datetime(pd.Series(tests['time_stamp']))

print(train.info())

train['Year'] = train['time_stamp'].apply(lambda x:x.year)
train['Month'] = train['time_stamp'].apply(lambda x: x.month)
train['weekday'] = train['time_stamp'].apply(lambda x: x.weekday())
train['time'] = train['time_stamp'].dt.time
tests['Year'] = tests['time_stamp'].apply(lambda x: x.year)
tests['Month'] = tests['time_stamp'].apply(lambda x: x.month)
tests['weekday'] = tests['time_stamp'].dt.dayofweek
tests['time'] = tests['time_stamp'].dt.time
train = train.drop('time_stamp', axis=1)
train = train.dropna(axis=0)
tests = tests.drop('time_stamp', axis=1)
tests = tests.fillna(method='pad')
for f in train.columns:
    if train[f].dtype=='object':
        if f != 'shop_id':
            print(f)
            lbl = preprocessing.LabelEncoder()
            train[f] = lbl.fit_transform(list(train[f].values))
for f in tests.columns:
    if tests[f].dtype == 'object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(tests[f].values))
        tests[f] = lbl.transform(list(tests[f].values))


feature_columns_to_use = ['Year', 'Month', 'weekday',
'time', 'longitude', 'latitude',
'wifi_id1', 'wifi_strong1', 'con_sta1',
 'wifi_id2', 'wifi_strong2', 'con_sta2',
'wifi_id3', 'wifi_strong3', 'con_sta3',
'wifi_id4', 'wifi_strong4', 'con_sta4',
'wifi_id5', 'wifi_strong5', 'con_sta5',
'wifi_id6', 'wifi_strong6', 'con_sta6',
'wifi_id7', 'wifi_strong7', 'con_sta7',
'wifi_id8', 'wifi_strong8', 'con_sta8',
'wifi_id9', 'wifi_strong9', 'con_sta9',
'wifi_id10', 'wifi_strong10', 'con_sta10',]

big_train = train[feature_columns_to_use]
big_test = tests[feature_columns_to_use]
train_X = big_train.values
test_X = big_test.values
train_y = train['shop_id']

gbm = xgb.XGBClassifier(silent=1, max_depth=10,
                    n_estimators=1000, learning_rate=0.05)
gbm.fit(train_X, train_y)
predictions = gbm.predict(test_X)

submission = pd.DataFrame({'row_id': tests['row_id'],
                            'shop_id': predictions})
print(submission)
submission.to_csv("submission.csv", index=False)
