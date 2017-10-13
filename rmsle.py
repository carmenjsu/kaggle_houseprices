import pandas as pd
import numpy as np
import csv
import sklearn
from sklearn.metrics import mean_squared_log_error
from sklearn.cross_validation import train_test_split
from math import sqrt

# checking rmse
actual_df = pd.read_csv('sample_submission.csv')
actual_y = actual_df.iloc[:,-1]
prediction_df = pd.read_csv('submissionv2.csv')
prediction_y = prediction_df.iloc[:,-1]
rmsle = np.sqrt(mean_squared_log_error(prediction_y, actual_y))
print ("rmsle")
print (rmsle)
