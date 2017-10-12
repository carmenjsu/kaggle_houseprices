'''
This script is created for the House Sale Price project on Kaggle
written by Carmen Su and run on Jupyter with %run 
'''


# data analysis and wrangling tools
import pandas as pd
import numpy as np

# visualization tools
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning tools 
from sklearn import linear_model
from sklearn.preprocessing import Imputer

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
print ("train_df")
print (train_df)
print (train_df.columns.values)
# train_df size: 1460 x 81, column 81 label "SalePrice"
# we want to seperate the label columne out first to avoid complecation 
train_df2 = train_df.iloc[:, :80]
print ("train_df2")
print (train_df2)
# train_df2 size: 1460 x 80
print ("test_df")
print (test_df)
# test_df size: 1459 x 80

#combine these two set for easy cleansing
frames = [train_df2, test_df]
df = pd.concat(frames)

# exam the data
print ("examing df")
print (df)
# now df contains 2919 x 80, top 1460 rows belong to trainning, last 1459 belong to testing
print (df.columns.values)

# analyse missing data
print ("percentage of missing data in each column")
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
dropout = missing_data[missing_data['Percent'] >0.005 ]
print ("dropout")
print (dropout)

# any features with missing data
# are to be removed from the training data. (for now)
df = df.drop(dropout.index,1)
df = df.fillna(0)

# then we check again what features the dataframe contains
print ("df after droping NaNs")
print (df.head())
print (df.count())
print (df.columns.values)
# it shows df is now 2919 x 62

print (df.dtypes)
# The trainning data contain categorical values, which is displayed as 'object'
# now, copy the object colums from the dataframe
catg_df = df.select_dtypes(include=['object']).copy()
print ('df categorical features')
print (catg_df.head())
print (catg_df.dtypes)
# it shows there are 28 categorical features
# we are now encoding thoses features one by one
# I know there might be a cleaver way of doing it
# Currently working on a do loop function to automate this part of the codes

print ('Examing each categorical features')

# feature 1-5
print (catg_df['BldgType'].value_counts())
print (catg_df['CentralAir'].value_counts())
print (catg_df['Condition1'].value_counts())
print (catg_df['Condition2'].value_counts())
print (catg_df['Electrical'].value_counts())

# feature 6-10
print (catg_df['ExterCond'].value_counts())
print (catg_df['ExterQual'].value_counts())
print (catg_df['Exterior1st'].value_counts())
print (catg_df['Exterior2nd'].value_counts())
print (catg_df['Foundation'].value_counts())

# feature 11-15
print (catg_df['Functional'].value_counts())
print (catg_df['Heating'].value_counts())
print (catg_df['HeatingQC'].value_counts())
print (catg_df['HouseStyle'].value_counts())
print (catg_df['KitchenQual'].value_counts())

# feature 16-20
print (catg_df['LandContour'].value_counts())
print (catg_df['LandSlope'].value_counts())
print (catg_df['LotConfig'].value_counts())
print (catg_df['LotShape'].value_counts())
print (catg_df['MSZoning'].value_counts())

# feature 21-28
print (catg_df['Neighborhood'].value_counts())
print (catg_df['PavedDrive'].value_counts())
print (catg_df['RoofMatl'].value_counts())
print (catg_df['RoofStyle'].value_counts())
print (catg_df['SaleCondition'].value_counts())
print (catg_df['SaleType'].value_counts())
print (catg_df['Street'].value_counts())
print (catg_df['Utilities'].value_counts())

# Find those vaules in the MAIN dataframes and replace with integer
cleanup_nums = {"Electrical":	{"Mix":1, "FuseP":2, "FuseF":3, "FuseA":4, "SBrkr":5},
				"MSZoning":     {"C (all)": 1, "RH": 2, "FV":3, "RM":4, "RL":5},
                "Street": 		{"Grvl": 1, "Pave": 2,},
                "LotShape":		{"IR3":1, "IR2":2, "IR1":3, "Reg":4},
                "LandContour": 	{"Low":1, "HLS":2, "Bnk":3, "Lvl":4},
                "Utilities": 	{"NoSeWa":1, "AllPub":2},
                "LotConfig": 	{"FR3":1, "FR2":2, "CulDSac":3, "Corner":4, "Inside":5},
                "LandSlope":	{"Sev":1, "Mod":2, "Gtl":3},
                "Neighborhood": {"Blueste":1, "NPkVill":2, "Veenker":3, "BrDale":4, "MeadowV":5,
                				"Blmngtn":6, "SWISU":7, "StoneBr":8, "ClearCr":9, "IDOTRR":10, 
                				"Timber":11, "NoRidge":12, "Mitchel":13, "Crawfor":14, "BrkSide":15, 
                				"SawyerW":16, "NWAmes":17, "Sawyer":18, "NridgHt":19, "Gilbert":20, 
                				"Somerst":21, "Edwards":22, "OldTown":23, "CollgCr":24, "NAmes":25},
                "Condition1":	{"RRNe":1, "RRNn":2, "PosA":3, "RRAe":4, "PosN":5,
                				"RRAn":6, "Artery":7, "Feedr":8, "Norm":9},
                "Condition2":	{"PosA":1, "RRAe":2, "RRAn":3, "RRNn":4, "Artery":5, 
                				"PosN":6, "Feedr":7, "Norm":8}, 
                "BldgType":		{"2fmCon":1, "Twnhs":2, "Duplex":3, "TwnhsE":4, "1Fam":5},
                "HouseStyle":	{"2.5Fin":1, "2.5Unf":2, "1.5Unf":3, "SFoyer":4, "SLvl":5,
                				"1.5Fin":6, "2Story":7, "1Story":8}, 
                "RoofStyle":	{"Shed":1, "Mansard":2, "Gambrel":3, "Flat":4, "Hip":5, "Gable":6},  
                "RoofMatl":		{"ClyTile":1, "Roll":2, "Membran":3, "Metal":4, "WdShake":5,
                				"WdShngl":6, "Tar&Grv":7, "CompShg":8}, 
                "Exterior1st":	{"ImStucc":1, "CBlock":2, "AsphShn":3, "Stone":4, "BrkComm":5,
                				"AsbShng":6, "Stucco":7, "WdShing":8, "BrkFace":9, "CemntBd":10,
                				"Plywood":11, "Wd Sdng": 12, "MetalSd":13, "HdBoard":14, "VinylSd":15},
                "Exterior2nd":	{"CBlock":1, "Other":2, "AsphShn":3, "Stone":4, "Brk Cmn":5,
                				"ImStucc":6, "AsbShng":7, "BrkFace":8, "Stucco":9, "Wd Shng":10, 
                				"CmentBd":11, "Plywood":12, "Wd Sdng":13, "HdBoard":14, "MetalSd":15, 
                				"VinylSd":16}, 
                "ExterQual":	{"Fa":1, "Ex":2, "Gd":3, "TA":4}, 
                "ExterCond":	{"Po":1, "Ex":2, "Fa":3, "Gd":4, "TA":5},
                "Foundation":	{"Wood":1, "Stone":2, "Slab":3, "BrkTil":4, "CBlock":5, "PConc":6}, 
                "Heating":		{"Floor":1, "OthW":2, "Wall":3, "Grav":4, "GasW":5, "GasA":6}, 
                "HeatingQC":	{"Po":1, "Fa":2, "Gd":3, "TA":4, "Ex":5}, 
                "CentralAir":	{"N":1, "Y":2}, 
                "KitchenQual":	{"Fa":1, "Ex":2, "Gd":3, "TA":4}, 
                "Functional":	{"Sev":1, "Maj2":2, "Maj1":3, "Mod":4, "Min1":5, 
                				"Min2":6, "Typ":7}, 
                "PavedDrive":	{"P":1, "N":2, "Y":3}, 
                "SaleType":		{"Con":1, "Oth":2, "CWD":3, "ConLw":4, "ConLI":5, 
                				"ConLD":6, "COD":7, "New":8, "WD":9}, 
                "SaleCondition":{"AdjLand":1, "Alloca":2, "Family":3, "Abnorml":4, "Partial":5, "Normal":6}
                }

catg_df.replace(cleanup_nums, inplace=True)
print ("catg_df")
print (catg_df.head())
# now we see all the categarical features have been converted, 2919 x 28
# replace them into the main data frame

df.replace(cleanup_nums, inplace=True)
print ("df after cleansing")
print (df.head())
# df is now size of 2919 x 62

# now we split df in to train_x, lable_y, and test_x 
train_x = df.iloc[:1460, :-1]
lable_y = train_df.iloc[:1460, -1]
test_x = df.iloc[1460:, :-1]

print ("train_x")
print (train_x)
# size 1460 x 61
print ("lable_y")
print (lable_y)
# size 1460 x 1
print ("test_x")
print (test_x)
# size 1459 x 61

# Train the model using the training data
regr = linear_model.LinearRegression()
regr.fit(train_x, lable_y)

pred_y = regr.predict(test_x)

print (pred_y)
print ("Coefficients: \n", regr.coef_)

print ('end')

