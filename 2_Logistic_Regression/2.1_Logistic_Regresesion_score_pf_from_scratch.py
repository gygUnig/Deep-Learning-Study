# Logistic Regression from scratch : score & pass/fail
# without using ML/DL Library

import numpy as np
import matplotlib.pyplot as plt


# data : score & pass/fail
score_ = [15, 24, 57, 78, 90, 114]
pf_ = [0, 0, 0, 1, 1, 1]

# zero for W_0, b_0
W_0 = 0
b_0 = 0


# sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))


# hypothesis function
def hypo_function(W,b,x):
    return sigmoid(W*x + b)


# cost function - BCE(Binary Cross Entropy)
def cost_function(W,b,x,y):

    sum = 0
    for i in range(len(x)):
        sum += -y[i] * np.log(hypo_function(W,b,x[i])) - (1-y[i]) * np.log(1-(hypo_function(W,b,x[i])))

    return sum / len(x)


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_w(W,b,x,y):

    sum_w = 0
    for i in range(len(x)):
        sum_w += y[i] * ((-x[i]*np.exp(-W*x[i]-b))/(1+np.exp(-W*x[i]-b))) + (1-y[i]) * ((x[i]*np.exp(W*x[i]+b))/(1+np.exp(W*x[i]+b)))

    return sum_w / len(x)


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W,b,x,y):

    sum_b = 0
    for i in range(len(x)):
        sum_b += y[i] * ((-np.exp(-W*x[i]-b))/(1+np.exp(-W*x[i]-b))) + (1-y[i]) * ((np.exp(W*x[i]+b))/(1+np.exp(W*x[i]+b)))

    return sum_b / len(x)


# final decide Pass/Fail function
def final(W,b,x):

    result = hypo_function(W,b,x)

    if result >= 0.5:
        pass_fail = 'pass'
    elif result < 0.5:
        pass_fail = 'fail'
    
    return pass_fail, result


# Learning rate & number of epoch
a = 0.01
n_epoch = 1000

# init w, b
w = W_0
b = b_0

# train
for epoch in range(n_epoch):
    w -= a * cost_diff_w(w,b,score_,pf_)
    b -= 100 * a * cost_diff_b(w,b,score_,pf_)

    if (epoch + 1) % 10 == 0:
        print("epoch {}/{} cost:{}".format(
            epoch+1, n_epoch, cost_function(w,b,score_,pf_)
        ))

# final decide pass/fail
for i in range(len(score_)):
    print('score:',score_[i], final(w,b,score_[i]))


# plot
x_plt = np.linspace(0,120,600)
y_plt = []

for i in range(len(x_plt)):
    _ , k = final(w,b,x_plt[i])
    y_plt.append(k)

plt.plot(x_plt, y_plt)
plt.scatter(score_,pf_)
plt.show()