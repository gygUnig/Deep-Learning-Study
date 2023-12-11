# Linear Regression from scratch
# without using ML/DL Library

import numpy as np
import matplotlib.pyplot as plt

# X, Y
X_data = [100, 125, 150, 190, 206]
Y_data = [105, 122, 155, 176, 207]

# random value for W_0, b_0
W_0 = np.random.uniform(-10,10)
b_0 = np.random.uniform(-10,10)


# hypothesis function
def hypo_function(W, b, x):
    return W * x + b


# cost function - MSE(Mean Squared Error)
def cost_function(W, b, x, y):

    sum = 0
    for i in range(len(x)):
        sum += (hypo_function(W, b, x[i]) - y[i]) ** 2
    return sum / len(x)


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_w(W, b, x, y):

    sum_w = 0
    for i in range(len(x)):
        sum_w += 2 * (hypo_function(W, b, x[i]) - y[i]) * x[i]
    return sum_w / len(x)


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W, b, x, y):

    sum_b = 0
    for i in range(len(x)):
        sum_b += 2 * (hypo_function(W,b,x[i]) - y[i]) 
    return sum_b / len(x)



# Learning rate & number of epoch
a = 0.00001
n_epoch = 100

# init w, b
w = W_0
b = b_0

# train & plot
for epoch in range(n_epoch):

    # gradiant descent
    w -= a * cost_diff_w(w, b, X_data, Y_data)
    b -= a * cost_diff_b(w, b, X_data, Y_data)

    print("epoch {}/{} w:{} b:{} cost:{}".format(
        epoch+1, n_epoch, w, b, cost_function(w, b, X_data, Y_data)
    ))


    x_plt = np.array([0,200])
    if epoch == 0:
        y_plt_1 = hypo_function(w,x_plt,b)
        plt.plot(x_plt, y_plt_1, label = f'epoch {epoch+1}')
    if epoch == 1:
        y_plt_3 = hypo_function(w,x_plt,b)
        plt.plot(x_plt, y_plt_3, label = f'epoch {epoch+1}')        
    if epoch == 2:
        y_plt_5 = hypo_function(w,x_plt,b)
        plt.plot(x_plt, y_plt_5, label = f'epoch {epoch+1}')        
    if epoch == 3:
        y_plt_10 = hypo_function(w,x_plt,b)
        plt.plot(x_plt, y_plt_10, label = f'epoch {epoch+1}')
    if epoch == 9:
        y_plt_100 = hypo_function(w,x_plt,b)
        plt.plot(x_plt, y_plt_100, label = f'epoch {epoch+1}')


plt.scatter(X_data, Y_data)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()