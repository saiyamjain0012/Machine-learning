import sys
import pandas
import matplotlib
import seaborn
import sklearn
import matplotlib.pyplot as plt
import seaborn as sms

from sklearn.model_selection import train_test_split
games=pandas.read_csv("games.csv")

#print the names of the coumns of the dataset

print(games.columns)
print(games.shape)


#Making a histogram of the average rating columns
plt.hist(games["average_rating"])
plt.show


#print first row of all the games with 0 rating
print(games[games["average_rating"]==0].iloc[0])


#print the first row of rest games
print(games[games["average_rating"]>0].iloc[0])


#Making dataset ready
games=games[games["users_rated"]>0]
games=games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show

#correlation matrix
corrmat=games.corr()
fig=plt.figure(figsize=(12,9))
sms.heatmap(corrmat, vmax=.8, square=True)
plt.show

#get all the columns from the dataset
columns=games.columns.tolist()

#taking only useful columns
columns=[c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#store the variables we'll be predicting on
target="average_rating"

#generating training and test data sets
train=games.sample(frac=0.8,random_state=1)
test=games.loc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)


#import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Initializing the regressor
LR=LinearRegression()

LR.fit(train[columns],train[target])

#Predicting on the test set
predictions=LR.predict(test[columns])

#Computing the error
mean_squared_error(predictions,test[target])

#Via Random Forest Model
from sklearn.ensemble import RandomForestRegressor

RFR=RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)

RFR.fit(train[columns],train[target])
RFRpredictions=RFR.predict(test[columns])

mean_squared_error(RFRpredictions,test[target])

test[columns].iloc[0]

#predicting with both models
rating_LR=LR.predict(test[columns].iloc[0].values.reshape(1,-1))

rating_RFR=RFR.predict(test[columns].iloc[0].values.reshape(1,-1))

print(rating_LR)

print(rating_RFR)


test[target].iloc[0]