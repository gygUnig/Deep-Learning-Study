# Softmax Regression from scratch : Zoo
# without using ML/DL Library

import numpy as np

# data load
data = np.loadtxt('../csv_datasets/data_zoo.csv',delimiter=',')  # (101, 17)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# train, test split
train_x = data[:80, :16]  # (80, 16)
train_y = data[:80, 16:]  # (80, 1)
test_x = data[80:, :16]  # (21, 16)
test_y = data[80:, 16:]  # (21, 1)


# number of class
num_of_class = len(np.unique(data[:,16:]))  # 7

# one hot encoding
max_train_y = int(np.max(train_y))  # 6
eye_train_y = np.eye(max_train_y + 1)  # (7, 7)
one_hot_train_y = eye_train_y[train_y.squeeze().astype(int)]  # (80,7)


# zero for W_0, b_0
W_0 = np.zeros((16,7))  # (16, 7)
b_0 = np.zeros((1,7))  # (1, 7)


# softmax function
def softmax_function(input_array):
    result_s = np.exp(input_array) / np.sum(np.exp(input_array), axis=1, keepdims=True)

    return result_s


# hypothesis function
def hypo_function(W,b,x_array):
    result_h = softmax_function(np.matmul(x_array,W) + b)

    return result_h


# cost function - Cross Entropy Loss Function
def cost_function(W,b,x_array,y_array):
    cost = (-np.sum(y_array * np.log(hypo_function(W,b,x_array)))) / x_array.shape[0]

    return cost


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_W(W,b,x_array,y_array):
    cost_w = np.dot(x_array.T, (hypo_function(W,b,x_array) - y_array)) / x_array.shape[0]

    return cost_w


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W,b,x_array,y_array):

    cost_b = np.sum(hypo_function(W,b,x_array) - y_array, axis=0, keepdims=True) / x_array.shape[0]

    return cost_b


# Learning rate & number of epoch
a = 0.1
n_epoch = 10000


# init W, b
w = W_0
b = b_0


# train
for epoch in range(n_epoch):

    w -= a * cost_diff_W(w,b,train_x,one_hot_train_y)
    b -= a * cost_diff_b(w,b,train_x,one_hot_train_y)

    if (epoch+1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost_function(w,b,train_x,one_hot_train_y)
        ))


# test
predict = hypo_function(w,b,test_x)
real = test_y

sum = 0
for p,r in zip(predict, real):
    if np.argmax(p) == int(r):
        sum += 1

# accuracy
accuracy = (sum / len(test_y)) * 100
print('test correct/entire = {}/{}'.format(sum, len(test_y)), 'accuracy:',accuracy)