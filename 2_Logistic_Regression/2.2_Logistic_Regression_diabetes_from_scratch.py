# Logistic Regression from scratch : diabetes
# without using ML/DL Library

import numpy as np

# data load
data = np.loadtxt('../Datasets/data_diabetes.csv', delimiter=',') # shape (759, 9)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# split train, test
train_x = data[:600, :8]  # (600, 8)
train_y = data[:600, 8:]  # (600, 1)
test_x = data[600:, :8]  # (159, 8)
test_y = data[600:, 8:]  # (159, 1)

# random value for W_0, b_0
W_0 = np.random.uniform(0,10,(8,1))  # (8, 1)
b_0 = np.random.uniform(0,10)


# sigmoid function
def sigmoid(x):
    
    return 1/(1+np.exp(-x))


# hypothesis function
def hypo_function(W_array, b, x_array):

    result = np.dot(x_array, W_array) + b

    return sigmoid(result)


# cost function - BCE(Binary Cross Entropy)
def cost_function(W_array, b, x_array, y_array):

    hypo = hypo_function(W_array, b, x_array)

    sum_cost = np.sum(-y_array * np.log(hypo) - (1 - y_array) * np.log(1 - hypo)) 

    return sum_cost


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_W(W_array, b, x_array, y_array):

    hypo = hypo_function(W_array, b, x_array)
    cost_w = np.dot(x_array.T, (hypo - y_array))

    return cost_w


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W_array, b, x_array, y_array):

    hypo = hypo_function(W_array, b, x_array)
    cost_b = np.sum(hypo - y_array)

    return cost_b

# Learning rate & number of epoch
a = 0.001
n_epoch = 1000

# init w, b 
w = W_0
b = b_0


# train
for epoch in range(n_epoch):

    w -= a*cost_diff_W(w,b,train_x,train_y)
    b -= a*cost_diff_b(w,b,train_x,train_y)

    if (epoch+1) % 10 == 0:
        print("epoch {}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost_function(w,b,train_x,train_y)
        ))


# test
test_predict = hypo_function(w,b,test_x)
predict = (test_predict >= 0.5).astype(int)


# accuracy
accuracy = np.mean(predict == test_y)
print('accuracy:', accuracy)

