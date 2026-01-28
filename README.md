# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, select the independent variable (R&D Spend) and dependent variable (Profit), and apply feature scaling.

2.Initialize the model parameters (weight, bias), learning rate, number of epochs, and loss list.

3.For each iteration, predict the output, compute the Mean Squared Error, calculate gradients, and update weight and bias using gradient descent.

4.Plot the loss curve and regression line, and display the final weight and bias values.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Latkshaya.s
RegisterNumber:  212225240078
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Load dataset
# -----------------------
data = pd.read_csv("ex3.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

# -----------------------
# Feature Scaling (IMPORTANT)
# -----------------------
x = (x - np.mean(x)) / np.std(x)

# -----------------------
# Parameters
# -----------------------
w = 0.0          # weight
b = 0.0          # bias
alpha = 0.01     # learning rate
epochs = 100
n = len(x)

losses = []

# -----------------------
# Gradient Descent
# -----------------------
for i in range(epochs):
    # Prediction
    y_hat = w * x + b

    # Loss (MSE)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

# -----------------------
# Plots
# -----------------------
plt.figure(figsize=(12, 5))

# Loss vs Iterations
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

# Regression Line
plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------
# Final Parameters
# -----------------------
print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:
![19AI410EX3](ex3.png)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
