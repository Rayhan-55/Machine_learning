#Importing the libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Training the Multiple Linear Regression Model on the Trainig set
from sklearn import linear_model
regressor=linear_model.LinearRegression()
regressor.fit(x_train,y_train) 

#predicting the Test set Results
y_pred=regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

""" what we did here is that we chaged it to colum so that we can see that was the actuall data and what our machine predicted 
 """

