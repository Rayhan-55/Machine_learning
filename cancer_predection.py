import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
upload=files.upload()

df=pd.read_csv('data.csv')
df.head()
#m for Malignant and B for Benning
 #for directly taking data file without downloading it to my pc
#kaggle api will be needed

''' #install the kaggle libary
! pip install kaggle
#make a directory named ".kaggle"
! mkdir -p ~/.kaggle
#copy the kaggle.json to the folder created
! cp kaggle.json ~/.kaggle/
#allocating the required permission for this file
! chmod 600 ~/.kaggle/kaggle.json '''

#Eda
#checking total number of rows and colums
df.shape

#check the colums and theri correspondig data types
#properties of data
df.info()

#second way to check for null values
df.isnull().sum()

#drop the column with missing values

df=df.dropna(axis=1)

df.shape

#checking after the droping missing values
df.dtypes

#data visulization
df['diagnosis'].value_counts()

sns.countplot(x='diagnosis', data=df,palette='Set2')

plt.xlabel('diagnosis')

#encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_Y=LabelEncoder()

#it is use to normailze neighbors label and it can also be used to transform categorical data to numeric data
#transforming the catagorical data to numeric
df.iloc[:,1]=labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.iloc[:,1].values
#first plotting
sns.pairplot(df.iloc[:,1:7],hue='diagnosis')

#corelation between colums
df.iloc[:,1:11].corr()

#heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(
    df.iloc[:, 1:12].corr()*100,
    cmap='viridis',
    annot=True,
    fmt='.2f'
)
plt.show()

#feature scaling
#split our data set into independent and dependent data sets
#indepnedent -->x
#depnedent-->y

# X এবং y তৈরি করা
x = df.iloc[:, 2:32].values
y = df.iloc[:, 1].values

# Label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0, stratify=y
)

# Model train
log_model, tree_model, forest_model = models(x_train, y_train)

x_test

def models(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # 1️⃣ Logistic Regression
    log = LogisticRegression(random_state=0, max_iter=1000)
    log.fit(x_train, y_train)

    # 2️⃣ Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(x_train, y_train)

    # 3️⃣ Random Forest
    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)

    # Training accuracy print করা
    print('[0] Logistic Regression Training Accuracy:', log.score(x_train, y_train))
    print('[1] Decision Tree Classifier Training Accuracy:', tree.score(x_train, y_train))
    print('[2] Random Forest Classifier Training Accuracy:', forest.score(x_train, y_train))

    return log, tree, forest

model = models(x_train,y_train)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example for Random Forest
y_pred = forest_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()
print(cm)
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
accuracy  = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall    = tp / (tp + fn)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range (len(model)):
  y_pred=model[i].predict(x_test)
  print("Model",i)
  print(accuracy_score(y_test,y_pred))
  print(classification_report(y_test,y_pred))

#predection
pred=log_model.predict(x_test)
print("our model predeciton: ")
print(pred)
print("actual values: ")
print(y_test)



