# Multi variable Linear Regression from scratch : test socre
# without using ML/DL Library


import numpy as np

# data load
data = np.loadtxt('../csv_datasets/data_test_score.csv', delimiter=',')  # shape (25, 4)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# split train, test
train_x = data[:20, :3]  # (20, 3)
train_y = data[:20, 3:]  # (20, 1)
test_x = data[20:, :3]  # (5, 3)
test_y = data[20:, 3:]  # (5, 1)

# random value for W_0, b_0
W_0 = np.random.uniform(0,10,(3,1))  # (3, 1)
b_0 = np.random.uniform(0,10)


# hypothesis function
def hypo_function(W_array, b, x_array):

    return np.dot(x_array, W_array) + b


# cost function - MSE(Mean Squared Error)
def cost_function(W_array, b, x_array, y_array):

    cost = np.sum( (hypo_function(W_array, b, x_array) - y_array) ** 2 ) / len(y_array)

    return cost


# cost differentiate function 1 - partial derivative with respect to w
def cost_diff_W(W_array, b, x_array, y_array):

    cost_W = np.dot(x_array.T, (hypo_function(W_array, b, x_array) - y_array)) * (2/len(y_array))

    return cost_W


# cost differentiate function 2 - partial derivative with respect to b
def cost_diff_b(W_array, b, x_array, y_array):

    cost_b = np.sum(hypo_function(W_array, b, x_array) - y_array) * (2/len(y_array))

    return cost_b


# Learning rate & number of epoch
a = 0.00001
n_epoch = 10000

# init w, b
w = W_0
b = b_0


# train
for epoch in range(n_epoch):
    w -= a*cost_diff_W(w,b,train_x,train_y)
    b -= a*cost_diff_b(w,b,train_x,train_y)

    if (epoch + 1) % 100 == 0:
        print("epoch {}/{} w:{} b:{} train_cost:{}".format(
            epoch+1, n_epoch, w, b, cost_function(w, b, train_x, train_y)
        ))


# test
predict = hypo_function(w, b, test_x)
real = test_y

for p,r in zip(predict, real):
    print('predict:',p,'real:',r)