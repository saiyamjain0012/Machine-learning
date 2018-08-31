import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sms
from sklearn.metrics import mean_squared_error
# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
gender_submission = pd.read_csv('gender_submission.csv')

#X = dataset.iloc[:, [2, 3]].values
#y = dataset.iloc[:, 4].values

mean=np.mean(train["Age"])
y=train["Survived"]


#filling mean age in no age values
train["Age"].fillna(mean,inplace=True)
 

#Taking only numeric columns
numeric_variables=list(train.dtypes[train.dtypes !="object"].index) 
del numeric_variables[0]
del numeric_variables[1]

#train[numeric_variables].head()
#train.describe()

test1=test[numeric_variables]
meantest=np.mean(test["Age"])
test["Age"].fillna(meantest,inplace=True)
test["Fare"].fillna(0,inplace=True)
from sklearn.ensemble import RandomForestRegressor

RFR=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)

RFR.fit(train[numeric_variables],y)
RFRpredictions=RFR.predict(test[numeric_variables])

mean_squared_error(RFRpredictions,gender_submission["Survived"])

