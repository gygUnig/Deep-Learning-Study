# Linear Regression from scratch 1 (No Bias)
# without using ML/DL Library

import numpy as np
import matplotlib.pyplot as plt

# data : weight & height
weight_ = [100, 125, 150, 190, 206]
height_ = [105, 122, 155, 176, 207]

# random value for W_0
W_0 = np.random.uniform(-10,10)


# hypothesis function
def hypo_function(W,x):
    return W*x

# cost function - MSE(Mean Squared Error)
def cost_function(W,x,y):

    sum = 0
    for i in range(len(x)):
        sum += (hypo_function(W,x[i]) - y[i]) ** 2

    return sum / len(x)

# cost differentiate function - partial derivative with respect to w
def cost_diff_function(W,x,y):

    sum_w = 0
    for i in range(len(x)):
        sum_w += 2 * (hypo_function(W,x[i]) - y[i]) * x[i]

    return sum_w / len(x)


# Learning rate & number of epoch
a = 0.00001
n_epoch = 50

# init w
w = W_0

# train & plot
for epoch in range(n_epoch):

    # gradiant descent
    w -= a*cost_diff_function(w,weight_,height_)

    print("epoch {}/{} w:{} train cost:{}".format(
        epoch+1, n_epoch, w, cost_function(w,weight_,height_)
    ))

    
    x_plt = np.array([0,200])
    if epoch == 0:
        y_plt_1 = hypo_function(w,x_plt)
        plt.plot(x_plt, y_plt_1, label = 'epoch 1')
    if epoch == 4:
        y_plt_5 = hypo_function(w,x_plt)
        plt.plot(x_plt, y_plt_5, label = 'epoch 5')
    if epoch == 9:
        y_plt_10 = hypo_function(w,x_plt)
        plt.plot(x_plt, y_plt_10, label = 'epoch 10')    
    if epoch == 49:
        y_plt_50 = hypo_function(w,x_plt)
        plt.plot(x_plt, y_plt_50, label = 'epoch 50')

plt.scatter(weight_, height_)
plt.xlabel('weight')
plt.ylabel('height')
plt.legend()
plt.show()