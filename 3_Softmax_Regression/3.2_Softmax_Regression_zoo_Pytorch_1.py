# Softmax Regression Practice : Zoo - Low level
# Using Pytorch

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# load data
data = np.loadtxt('../Datasets/data_zoo.csv', delimiter=',')  # (101, 17)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# numpy array to tensor
data = torch.tensor(data, dtype=torch.float32)  # torch.Size([101, 17])

# number of class
num_class = len(torch.unique(data[:, 16:]))  # 7

# split train data, test data
train_x = data[:80, :16]  # torch.Size([80, 16])
train_y = data[:80, 16:]  # torch.Size([80, 1])
test_x = data[80:, :16]  # torch.Size([21, 16])
test_y = data[80:, 16:]  # torch.Size([21, 1])

# one hot encoding
train_y_one_hot = torch.zeros(80, 7)
train_y_one_hot.scatter_(1, train_y.long(), 1)  # scatter_(dimension(int), index(LongTensor), value)


# init W, b
W = torch.zeros((16,7), requires_grad=True)
b = torch.zeros((1,7), requires_grad=True)

# optimizer
optimizer = optim.SGD([W,b], lr = 0.01)

# number of epoch
n_epoch = 10000

# train
for epoch in range(n_epoch):

    # hypothesis function
    hypo = F.softmax(train_x.matmul(W)+b, dim = 1)

    # cost function
    cost = (train_y_one_hot * -torch.log(hypo)).sum(dim=1).mean()

    optimizer.zero_grad()  # reset gradiants
    cost.backward()  # backpropagate errors
    optimizer.step()  # update model parameters

    if (epoch+1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost.item()
        ))


# test & accuracy
predict = F.softmax(test_x.matmul(W)+b, dim=1)
predict = torch.argmax(predict, dim=1)
real = test_y.squeeze()

for p,r in zip(predict, real):
    print("test Predict:",p.item(), "Real:",r.item())

correct = (predict==real).sum().item()

print('test correct/entire = {}/{},'.format(correct, len(real)),'accuracy :',100*correct/len(real) )