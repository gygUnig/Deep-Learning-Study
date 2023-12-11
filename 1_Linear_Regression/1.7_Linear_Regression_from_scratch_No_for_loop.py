# Linear Regression from scratch 3 (No for loop for cost function, cost diff function (Use vectorization))
# without using ML/DL Library


import numpy as np
import matplotlib.pyplot as plt

# data : weight & height
weight_ = np.array([100, 125, 150, 190, 206])
height_ = np.array([105, 122, 155, 176, 207])

# init w, b
w = 0
b = 0

# hypothesis function
def hypo_function(W,b,x):
    return W*x + b


# cost function - MSE(Mean Squared Error)
def cost_function(W,b,x,y):
    
    result = np.sum(np.square(hypo_function(W,b,x) - y))
    return result / len(x)


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_w(W,b,x,y):
    
    result = 2 * np.sum((hypo_function(W,b,x) - y) * x)
    return result / len(x)


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W,b,x,y):
    
    result = 2 * np.sum((hypo_function(W,b,x) - y))
    return result / len(x)

# Learning rate & number of epoch
a = 0.00001
n_epoch = 100


# train & plot
for epoch in range(n_epoch):
    
    # gradient descent
    w -= a * cost_diff_w(w,b,weight_,height_)
    b -= a * cost_diff_b(w,b,weight_,height_)
    
    print("epoch {}/{} w:{}, b:{}, cost:{}".format(
        epoch+1, n_epoch, w, b, cost_function(w, b, weight_, height_)
    ))
    
    x_plt = np.array([0, 200])
    if epoch == 0:
        y_plt_1 = hypo_function(w, b, x_plt)
        plt.plot(x_plt, y_plt_1, label = 'epoch 1')
    elif epoch == 2:
        y_plt_3 = hypo_function(w, b, x_plt)
        plt.plot(x_plt, y_plt_3, label = 'epoch 3')
    elif epoch == 4:
        y_plt_5 = hypo_function(w, b, x_plt)
        plt.plot(x_plt, y_plt_5, label = 'epoch 5')
    elif epoch == 9:
        y_plt_10 = hypo_function(w, b, x_plt)
        plt.plot(x_plt, y_plt_10, label = 'epoch 10')
    elif epoch == 99:
        y_plt_100 = hypo_function(w, b, x_plt)
        plt.plot(x_plt, y_plt_100, label = 'epoch 100')

plt.scatter(weight_, height_)
plt.xlabel('weight')
plt.ylabel('height')
plt.legend()
plt.show()