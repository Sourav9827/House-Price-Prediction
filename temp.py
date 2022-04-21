# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:26:31 2021

@author: sRv
"""



train_copy = train.copy() 

for i in cols_object:
    if train[i].isnull().sum() > 0:
        train_copy[i].fillna("Missing", inplace = True)

for i in cols_num:
    if train[i].isnull().sum() > 0:
        train_copy[i].fillna(train[i].median(), inplace = True)
 

train_copy_null_sum = train_copy.isnull().sum()
sns.heatmap(train_copy.isnull(),yticklabels=False,cbar=False)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    train_copy[feature]=train_copy['YrSold']-train_copy[feature]

num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    train_copy[feature]=np.log(train_copy[feature])
    
for feature in cols_object:
    temp=train_copy.groupby(feature)['SalePrice'].count()/len(train_copy)
    temp_df=temp[temp>0.01].index
    train_copy[feature]=np.where(train_copy[feature].isin(temp_df),train_copy[feature],'Rare_var')
    
for feature in cols_object:
    labels_ordered=train_copy.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    train_copy[feature]=train_copy[feature].map(labels_ordered)
    
scaling_feature=[feature for feature in train_copy.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature) 

feature_scale=[feature for feature in train_copy.columns if feature not in ['Id','SalePrice']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(train_copy[feature_scale])
 
scaler.transform(train_copy[feature_scale])

data = pd.concat([train_copy[['Id', 'SalePrice']].reset_index(drop=True),
pd.DataFrame(scaler.transform(train_copy[feature_scale]), columns=feature_scale)],axis=1)









