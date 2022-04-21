# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:18:04 2021

@author: sRv
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import statsmodels.api as s
from statsmodels.formula.api import ols
from scipy import stats
from datetime import date
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from xgboost import XGBRegressor

os.chdir("D:\Temp\House Price")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_copy = train.copy()
 
train_null_sum = train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
plt.show()
test_null_sum = test.isnull().sum()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
plt.show()

train.columns
test.columns

cols_object = []
for i in train.columns:
    if train[i].dtypes == 'O':
        cols_object.append(i)   

for i in cols_object:
    if train[i].isnull().sum() > 0:
        print(i,"\t", train[i].isnull().mean() * 100,"% data is missing")
        
#for i in range(0,len(cols_object)):
   # print('\n\n',train[cols_object[i]].value_counts(dropna=False))
        
train.duplicated().sum()
test.duplicated().sum()

cols_num = []
for i in train.columns:
    if train[i].dtypes == 'int64' or train[i].dtypes == 'float64':
        cols_num.append(i)

for i in cols_num:
    if train[i].isnull().sum() > 0:
        print(i,"\t", train[i].isnull().mean() * 100, "% data is missing")
         
#for i in cols_object:
    #sns.boxplot(i, 'LotFrontage', data = train)
    #plt.show()
#sns.boxplot('PavedDrive', 'LotFrontage', data = train)
#train.pivot_table(index = 'PavedDrive', values = 'LotFrontage', aggfunc = 'median')
#sns.boxplot('TotRmsAbvGrd', 'LotFrontage', data = train)
train.pivot_table(index = 'TotRmsAbvGrd', values = 'LotFrontage', aggfunc = 'median')
#sns.boxplot('Fireplaces', 'LotFrontage', data = train)
#train.pivot_table(index = 'Fireplaces', values = 'LotFrontage', aggfunc = 'median')

def fill_LotFrontage(c):
    TRAG = c[0]
    LFA = c[1]
    if pd.isnull(LFA):
        if TRAG == "2":
            LFA = 50.0
        elif TRAG == "3":
            LFA = 28.5
        elif TRAG == "4":
            LFA = 56
        elif TRAG == "5":
            LFA = 60
        elif TRAG == "6":
            LFA = 69
        elif TRAG == "7":
            LFA = 72
        elif TRAG == "8":
            LFA = 76
        elif TRAG == "9":
            LFA = 85
        elif TRAG == "10":
            LFA = 85
        elif TRAG == "11":
            LFA = 85
        elif TRAG == "12":
            LFA = 90
        else:
            LFA = 60
    return LFA
train['LotFrontage'] = train[['TotRmsAbvGrd','LotFrontage']].apply(fill_LotFrontage,axis=1)
train['GarageYrBlt'].describe()
train['GarageYrBlt'].fillna(0, inplace = True)
train['MasVnrArea'].fillna(0, inplace = True)

for i in cols_object:
    if train[i].isnull().sum() > 0:
        train[i].fillna("Missing", inplace = True)

train_summary = train.describe()
test_summary = test.describe()

train.info()
test.info()

cor = train.corr()

for i in cols_num:
    sns.lineplot(x = np.log(train[i]), y = train['SalePrice'])
    plt.show()
    
for i in cols_object:
    sns.barplot(x = train[i], y = train['SalePrice'])
    plt.show()
    
train_cleaned = pd.get_dummies(train, columns =  cols_object)
train_cleaned_null_sum = train_cleaned.isnull().sum()

train_cleaned_info = train_cleaned.dtypes

for i in train_cleaned.columns:
    sns.regplot(x = train_cleaned[i], y = train_cleaned['SalePrice'])
    plt.show()

null_cols = []
for i in cols_object:
    if train_copy[i].isnull().sum() > 0:
        null_cols.append(i)
str(null_cols)

Missing = ['Alley_Missing', 'MasVnrType_Missing', 'BsmtQual_Missing', 'BsmtCond_Missing', 'BsmtExposure_Missing', 'BsmtFinType1_Missing', 'BsmtFinType2_Missing', 'Electrical_Missing', 'FireplaceQu_Missing', 'GarageType_Missing', 'GarageFinish_Missing', 'GarageQual_Missing', 'GarageCond_Missing', 'PoolQC_Missing', 'Fence_Missing', 'MiscFeature_Missing']

train_cleaned["Missing_sum"] = 0
for i in range(0,len(train_cleaned),1):
    k = 0
    for j in Missing:
       k += train_cleaned[j][i]
    train_cleaned["Missing_sum"][i] = k   

temp = train_cleaned[Missing]

sns.regplot(train_cleaned["Missing_sum"], train_cleaned["SalePrice"])

cor_cleaned = train_cleaned.corr()
train_cleaned['SalePrice'].plot()

for i in cols_num:
    train_cleaned[i] = np.log(train_cleaned)

x = train_cleaned.drop(columns = ['Id','SalePrice'])
y = train_cleaned['SalePrice']
y = np.log(y)

#OLS Model

lm = s.OLS(y,x).fit()
lm.summary()

lm_modified = s.OLS(y,x[['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','TotalBsmtSF','GrLivArea','BsmtFullBath','GarageArea','WoodDeckSF','ScreenPorch','MSZoning_FV','MSZoning_RL','Neighborhood_Crawfor','Neighborhood_NridgHt','Neighborhood_StoneBr','Condition1_Norm','Condition2_PosN','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl','Exterior1st_BrkFace','Functional_Min2','Functional_Typ']]).fit()
lm_modified.summary()

x = x[['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','TotalBsmtSF','GrLivArea','BsmtFullBath','GarageArea','WoodDeckSF','ScreenPorch','MSZoning_FV','MSZoning_RL','Neighborhood_Crawfor','Neighborhood_NridgHt','Neighborhood_StoneBr','Condition1_Norm','Condition2_PosN','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl','Exterior1st_BrkFace','Functional_Min2','Functional_Typ']]

#Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
print("\n\n Shapes Of Training And Test Data:\n",x_train.shape, x_test.shape, y_train.shape, y_test.shape)
base_pred = np.mean(y_test)
print("Base Prediction:\t",base_pred)
base_pred = np.repeat(base_pred, len(y_test))
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print("RMSE:\t",base_root_mean_square_error)

#Random Forest 

rf = RandomForestRegressor(random_state=10)
model_rf = rf.fit(x_train, y_train)
prediction_rf = rf.predict(x_test)

rf_mse = mean_squared_error(y_test, prediction_rf)
rf_rmse = np.sqrt(rf_mse)
print("\n RMSE Of Random Forest Model: \t",rf_rmse)

r2_rf_test = model_rf.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_rf_test)
r2_rf_train = model_rf.score(x_train, y_train)
print("\n R-Squared value Of Test Data: \t", r2_rf_train)

residuals_rf=y_test-prediction_rf
sns.regplot(x=prediction_rf, y=residuals_rf, scatter=True, fit_reg=True)
plt.title("Residuals For Random Forest Model")
plt.show()

#Linear Regression

lnrg = LinearRegression(fit_intercept=True)
model_lnrg = lnrg.fit(x_train, y_train)
prediction_lnrg = lnrg.predict(x_test)

lnrg_mse = mean_squared_error(y_test, prediction_lnrg)
lnrg_rmse = np.sqrt(lnrg_mse)
print("\n RMSE Of Linear Regression Model: \t",lnrg_rmse)

r2_lnrg_test = model_lnrg.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_lnrg_test)
r2_lnrg_train = model_lnrg.score(x_train, y_train)
print("\n R-Squared value Of Train Data: \t", r2_lnrg_train)

residuals_lngr=y_test-prediction_lnrg
sns.regplot(x=prediction_lnrg, y=residuals_lngr, scatter=True, fit_reg=True)
plt.title("Residuals For Linear Regression Model")
plt.show()

#Decision Tree

tree = DecisionTreeRegressor()
model_tree = tree.fit(x_train, y_train)
prediction_tree = tree.predict(x_test)

tree_model_mse = mean_squared_error(y_test, prediction_tree)
tree_model_rmse = np.sqrt(tree_model_mse)
print("\n\n RMSE Of Decision Tree Model:\t",tree_model_rmse)

r2_tree_test = model_tree.score(x_test, y_test)
print("\n R-Squared value Of Test Data: \t",r2_tree_test)
r2_tree_train = model_tree.score(x_train, y_train)
print("\n R-Squared value Of Train Data: \t", r2_tree_train)

residuals_tree=y_test-prediction_tree
sns.regplot(x=prediction_tree, y=residuals_tree, scatter=True, fit_reg=True)
plt.title("Residuals For Decision Tree Model")
plt.show()


# RMSE of Linear Regression Is The Least Among All three.
# Test Predictions Using Linear Regression.

test_cleaned = pd.get_dummies(test, columns = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'])

x.columns
test_cleaned['RoofMatl_Metal'] = 0
test_cleaned['RoofMatl_Roll'] = 0
test_cleaned['RoofMatl_Membran'] = 0
test_cleaned.isnull().sum()
test_cleaned['MasVnrArea'].fillna(test_cleaned['MasVnrArea'].mean(), inplace = True)
test_cleaned['BsmtFinSF1'].fillna(test_cleaned['BsmtFinSF1'].mean(), inplace = True)
test_cleaned['TotalBsmtSF'].fillna(test_cleaned['TotalBsmtSF'].mean(), inplace = True)
test_cleaned['BsmtFullBath'].fillna(test_cleaned['BsmtFullBath'].mean(), inplace = True)
test_cleaned['GarageArea'].fillna(test_cleaned['GarageArea'].mean(), inplace = True)


x2 = test_cleaned[['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','TotalBsmtSF','GrLivArea','BsmtFullBath','GarageArea','WoodDeckSF','ScreenPorch','MSZoning_FV','MSZoning_RL','Neighborhood_Crawfor','Neighborhood_NridgHt','Neighborhood_StoneBr','Condition1_Norm','Condition2_PosN','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl','Exterior1st_BrkFace','Functional_Min2','Functional_Typ']]
x2.isnull().sum()

print(x.shape, '\t', x2.shape)

lnrg_train = LinearRegression(fit_intercept=True)
model_test = lnrg_train.fit(x,y)
prediction_test = lnrg.predict(x2)

prediction_test_data = pd.DataFrame(test["Id"])
prediction_test_data["SalePrice"] = np.exp(prediction_test)

prediction_test_data.to_csv(r"D:\Temp\\House Price\Final_Submission.csv", index=False)



















