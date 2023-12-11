# Logistic Regression Practice : diabetes
# Using Pytorch

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# load data
data = np.loadtxt('../Datasets/data_diabetes.csv',delimiter=',')  # (759, 9)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# numpy array to torch tensor
data = torch.tensor(data, dtype=torch.float32)

# split train data, test data
train_x = data[:600, :8]  # torch.Size([600, 8])
train_y = data[:600, 8:]  # torch.Size([600, 1])
test_x = data[600:, :8]  # torch.Size([159, 8])
test_y = data[600:, 8:]  # torch.Size([159, 1])

# init W, b
W = torch.zeros((8,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W,b], lr = 0.01)

# number of epoch
n_epoch = 10000


# train
for epoch in range(n_epoch):

    # hypothesis function
    hypo = torch.sigmoid(train_x.matmul(W) + b)

    # cost function
    cost = F.binary_cross_entropy(hypo, train_y)

    optimizer.zero_grad()  # reset gradiants
    cost.backward()  # backpropagate errors
    optimizer.step()  # update model parameters

    if (epoch + 1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost.item()
        ))

# test & accuracy
predict = torch.sigmoid(test_x.matmul(W) + b)
predict = torch.round(predict)

correct = (predict == test_y).sum().item()

print("test accuracy :", (correct/len(test_y))*100)