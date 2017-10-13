'''
This script is created for the House Sale Price project on Kaggle
written by Carmen Su and run on cmd
This is a version 2 of the script
Dropping columnes with 45%+ missing data and replace NaNs with means
Use dummies for catagorical datas
'''


# data analysis and wrangling tools
import pandas as pd
import numpy as np
import csv

# visualization tools
import matplotlib.pyplot as plt

# machine learning tools and validation tools
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from math import sqrt

# input data files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

"""
Author has decisded to use Linear Regression for this use case, Ax=b
preparing datasets for Linear Regression model y = Xb + e, 
where y is the target values, X is the training data matrix, 
b is the coefficient vector, and e is the noise.
from the training data we have y and X to train b and e
Now we are preparing the data from y and X from train_df
"""
# first of all, exame the data
# we want to seperate the ID and label columnes out first to avoid complecation 
lable_y = train_df.iloc[:, -1]
ID = test_df.iloc[:,0]
train_df = train_df.iloc[:, 1:-1]
print ("train_df")
print (train_df)
# train_df size: 1460 x 79, up to SaleCond
test_df = test_df.iloc[:, 1:]
print ("test_df")
print (test_df)
# test_df size: 1459 x 79, up to SaleCond

#combine these two set for easy cleansing
frames = [train_df, test_df]
df = pd.concat(frames)

# exam the data
print ("examing df")
print (df)
# now df contains 2919 x 79, top 1460 rows belong to trainning, last 1459 belong to testing
print (df.columns.values)

# creating dummies for categorical features 
df = pd.get_dummies(df)

# analyse missing data
print ("percentage of missing data in each column")
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print (missing_data)

# drop anything above 45%
dropout = missing_data[missing_data['Percent'] >0.45]
df = df.drop(dropout.index,1)

# fill other NaNs with mean
df = df.fillna(df.mean())
# then we check again what features the dataframe contains
print ("df after dropping and filling NaNs with means")
print (df.head())
print (df.columns.values)
# it shows df is now 2919 x 288 cuz of dummies

# now we split df in to train_x, lable_y, and test_x 
train_x = df.iloc[:1460, :]
test_x = df.iloc[1460:, :]


print ("train_x")
print (train_x)
# size 1460 x 288
print ("lable_y") # lable_y was extracted very early on
print (lable_y)
# size 1460 x 1
print ("test_x")
print (test_x)
# size 1459 x 288

print ("training model")
# Train the model using the training data
regr = linear_model.LinearRegression()
regr.fit(train_x, lable_y)

print ("predicting pred_y")
pred_y = regr.predict(test_x)
print (pred_y)
print (pred_y.shape)
# pred_y is a numpy.ndarray
pred_y = pred_y.reshape((-1,1))
print (pred_y.shape)
# now size: 1459 x 1, numpy.ndarry
prediction_y = pd.DataFrame(pred_y)
print (prediction_y)

#combine ID and prediction_y for submission
print (ID) # ID was extracted very early on
submission = pd.concat([ID, prediction_y], axis=1)
print ("submission")
submission.columns = ["ID", "SalePrice"]
print (submission.columns.values)
print (submission)


submission.to_csv('submissionv2.csv')

print ('end')
