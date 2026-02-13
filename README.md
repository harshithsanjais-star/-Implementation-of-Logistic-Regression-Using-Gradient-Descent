# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/acer/Downloads/Placement_Data (2).csv")

data1=data.copy()
data1=data1.drop(['sl_no','salary'],axis=1)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])

x = data1.iloc[:, :-1].values
y = data1["status"].values

theta = np.random.randn(x.shape[1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,x)
accuracy=np.mean(y_pred==y)

print("Accuracy:",accuracy)
```

## Output:
"C:\Users\acer\Pictures\Screenshots\Screenshot (38).png"


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

