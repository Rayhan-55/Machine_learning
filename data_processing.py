import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
data_set=pd.read_csv('Data.csv') 
x=data_set.iloc[:,:-1].values #matrix of feature
y=data_set.iloc[:,-1].values#dependent variable of vector

#print(x)

#print(y)
#Taking care of missing data
from sklearn import impute
from sklearn.impute import SimpleImputer
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
impute.fit(x[:,1:3])
x[:,1:3]=impute.transform(x[:,1:3])#transform update kore return kore
#Encoding catagorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x=np.array(ct.fit_transform(x))
print(x)

#encoding independent variables

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)