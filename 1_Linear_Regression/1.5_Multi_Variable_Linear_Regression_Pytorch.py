# Multi variable Linear Regression practice : test socre
# using Pytorch

import numpy as np
import torch
import torch.optim as optim


# data load
data = np.loadtxt('../csv_datasets/data_test_score.csv', delimiter=',')

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# numpy array to torch tensor
data = torch.tensor(data, dtype=torch.float32)  # torch.Size([25, 4])

# split train data, test data
train_x = data[:20, :3]  # torch.Size([20, 3])
train_y = data[:20, 3:]  # torch.Size([20, 1])

test_x = data[20:, :3]  # torch.Size([5, 3])
test_y = data[20:, 3:]  # torch.Size([5, 1])


# init W, b
w = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# optimizer
optimizer = optim.SGD([w,b], lr = 0.00001)

# number of epoch
n_epoch = 10000

# train
for epoch in range(n_epoch):

    # hypothesis function
    hypo = train_x.matmul(w) + b

    # cost function
    cost = torch.mean((hypo - train_y) ** 2)

    optimizer.zero_grad()  # reset gradients
    cost.backward()  # backpropagate errors
    optimizer.step()  # update model parameters

    if (epoch + 1) % 100 == 0:
        print('epoch:{}/{} train_cost:{}'.format(
            epoch+1, n_epoch, cost.item()
        ))

# test
predict = test_x.matmul(w) + b
print("Predict :", predict.squeeze())
print("real    :", test_y.squeeze())