

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Startv the program.
2.import numpy as np. 
3.Give the header to the data.
4.Find the profit of population.
5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6.End the program.

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
1.profit prediction

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/7fb32de5-c945-4359-be9b-32c9a8ade66f)

2.function output

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/4ce7c8d9-28b0-42e5-bf7f-2a828a0c760f)

3.Gradient Descent

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/ab6e2497-bb79-431b-82d7-c3da4d727dfd)

4.Cost function using gradient descent

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/d4b602d2-48fd-4b08-97cb-1401cf2e8cd3)

5.Linear regression using profit prediction

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/48e705d5-d941-4b93-8a79-91ab8c09aa78)

6.Profit prediction for a population of 35,000

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/43148c82-fc3a-4345-86aa-47edb3bbe107)

7.Profit prediction for a population of 70,000

![image](https://github.com/Tharun-1000/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/135952958/d5429179-4770-4759-a7c0-b9221ee5d9bd)

















## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
