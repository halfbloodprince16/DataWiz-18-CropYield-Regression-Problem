"""
Created on Sun Sep 16 23:16:35 2018

@author: hbp16
"""
import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')

import matplotlib.pyplot as plt
x = df['Year']
y = df['Yield']

plt.scatter(x,y)
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['District'] = le.fit_transform(df['District'])
dummies = pd.get_dummies(df['District'])
df = pd.concat([df,dummies],axis=1)

df['Season'] = le.fit_transform(df['Season'])
df['Crop'] = le.fit_transform(df['Crop'])

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df.iloc[:,[3]] = mms.fit_transform(df.iloc[:,[3]])

#fill up missing values

'''
AvgAirTemp       float64
MinAirTemp       float64
MaxAirTemp       float64
AvgTempSkew      float64
AvgTempKurt      float64
AvgRelHum        float64
AvgRelHumSkew    float64
AvgRelHumKurt    float64
AvgDewPt         float64
AvgDewPtSkew     float64
AvgDewPtKurt     float64
AvgPrec          float64
AvgPrecSkew      float64
AvgPrecKurt      float64
AvgWind          float64
'''
from sklearn.preprocessing import Imputer
imp = Imputer()
df.iloc[:,[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]] = imp.fit_transform(df.iloc[:,[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])

X = df.iloc[:,1:20]
Y = df.iloc[:,[20]]

from sklearn.feature_selection import SelectKBest,chi2
model = SelectKBest(chi2,k=5).fit(X,Y)
score = model.scores_

import seaborn as sns
sns.heatmap(df.corr(),annot=True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=13)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr = lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state=0)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=70,random_state=0)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

from sklearn.svm import SVR
svr = SVR(kernel='rbf',degree=3)
svr = svr.fit(X_train,y_train)
y_pred = svr.predict(X_test)


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=13, loss='ls')
gbr = gbr.fit(X_train,y_train)
y_pred = gbr.predict(X_test)


from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor(n_estimators=100)
br = br.fit(X_train,y_train)
y_pred = br.predict(X_test)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,y_pred)))





