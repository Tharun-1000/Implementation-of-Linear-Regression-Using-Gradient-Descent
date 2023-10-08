# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your compiler.
2. Type the required program.
3. Print the program.
4.End the program.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: tharun k
RegisterNumber:212222040172
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1 (2).txt", header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Popuation of city (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  m=len(y)
  h=X.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="purple")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions = np.dot(theta.transpose(),x)
    return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000 , we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000 , we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![273454130-b516bb5c-f151-429d-bf50-75dbdd9e734a](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/5217ab44-3639-492b-9875-1404d3c92366)
![273454288-ce1dcc68-c409-4861-93e6-f229f561f04f](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/32297068-a5ec-4fbe-b881-9e254a1cfcba)
![273454297-6e93a706-8ec6-44dc-8b83-d0e863191ca2](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/9fb95559-4f32-477f-9935-728a39c02ff6)
![273454305-1fda2603-8672-4fcf-a079-953fb5295842](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/d8b64dc6-05b0-4288-92ef-cf28822a9307)
![273454313-fdcf5fa4-096f-40ce-8d41-e056db1efe41](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/b32ebfd1-ceb6-49a4-994d-4f6983f7b327)
![273454322-d55826b0-a39c-43c7-a491-fa08f39bb9a5](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/9161cec4-df35-4071-92b5-f352c14021c4)
![273454330-dbbae8c9-90ad-4d31-9ba3-23f59f1238a3](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/8eee44c7-d289-42ae-af4e-fbc1a6e3564c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
