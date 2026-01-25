#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
from typing import ValuesView
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values
#Training the Decision Tree Regression Model

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
#Predicting a new result

regressor.predict([[6.5]])

#Visualising the Decision Tree Regression results(higher resolution)

X_grid=np.arange(min(x),max(x),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(x,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff(Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()