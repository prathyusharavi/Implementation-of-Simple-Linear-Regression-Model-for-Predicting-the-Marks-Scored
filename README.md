I# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed package
2. Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the value
5.end the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YENUGANTI PRATHYUSHA
RegisterNumber: 212223240187 
*/
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:
#head:
![image](https://github.com/user-attachments/assets/fce490e4-d40d-43c8-8b04-dd41c3308c26)

#tail:
![image](https://github.com/user-attachments/assets/131e5799-e798-43e5-bccb-6e7c69e7fcca)

#Array value of x:
![image](https://github.com/user-attachments/assets/235dcd38-07e9-4456-890e-ae3549b6cc7f)

#Array value of Y:
![image](https://github.com/user-attachments/assets/55d93109-0e77-4120-95ba-88aa30f54418)

#Y prediction:
![image](https://github.com/user-attachments/assets/11d73075-8da5-4cf9-be94-ff460e216a9d)

#Training set graph:
![image](https://github.com/user-attachments/assets/86456f15-e8ce-45ad-98ba-475648531fbe)

#testing set graph:
![image](https://github.com/user-attachments/assets/b709f4c4-e4cd-41ef-bf67-5f7e1f346a73)

#Values of MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/4deea0da-68d1-49c2-b962-09d1815ac96d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
