#Importing Libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing data set
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Spliting the dataset into the training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Training the simple Linear regression model on the Training
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test result
y_pred=regressor.predict(x_test)

#Visualising the Training set result
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience(Training set)')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set result
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience(Test set)')
plt.ylabel('Salary')
plt.show()

#extra
print(regressor.predict([[15]]))

""" Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

12→scalar

[12]→1D array

[[12]]→2D array """

