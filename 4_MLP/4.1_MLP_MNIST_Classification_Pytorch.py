# MNIST Classification practice : Fully Connected Neural Network
# Using Pytorch


# Use batch size
# Use normalization
# find the epoch at which the slope of the cost_test graph changes from negative to positive
# plot loss-epoch graph

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Start")


# train, test data load
data = np.loadtxt("../csv_datasets/data_mnist_train.csv", delimiter=',')  # (60000, 785)
test_data = np.loadtxt("../csv_datasets/data_mnist_test.csv", delimiter=',')  # (10000, 785)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)
np.random.shuffle(test_data)

# numpy array to torch tensor
data = torch.tensor(data, dtype=torch.float32)  # torch.Size([60000, 785])
test_data = torch.tensor(test_data, dtype=torch.float32)  # torch.Size([10000, 785])

# split x,y
train_x = data[:, 1:]  # torch.Size([60000, 784])
train_y = data[:, :1].long()  # torch.Size([60000, 1])

test_x = test_data[:, 1:]  # torch.Size([10000, 784])
test_y = test_data[:, :1].long()  # torch.Size([10000, 1])

# x - normalization (mean max scale) : X = (x - x_min)/(x_max - x_min)
# MNIST data : 0 ~ 255
train_x = train_x / 255
test_x = test_x / 255

# model
model = nn.Sequential(
    nn.Linear(784,64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,10)
)

# optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.0003)

# batch size
batch_size = 3000

# number of epoch
n_epoch = 1000


plt_epoch = []
plt_cost_train = []
plt_cost_test = []
# train
for epoch in range(n_epoch):

    for b in range(int(len(train_y)/batch_size)):

        # cost
        cost = nn.CrossEntropyLoss()(model(train_x[batch_size*b:batch_size*(b+1),:]), train_y[batch_size*b:batch_size*(b+1),:].squeeze())

        optimizer.zero_grad()  # reset gradiants
        cost.backward()  # backpropagate errors
        optimizer.step()  # update model parameters

    with torch.no_grad():
        test_cost = nn.CrossEntropyLoss()(model(test_x),test_y.squeeze())

    if (epoch + 1) % 100 == 0:
        print("epoch {}/{} train_cost:{} test_cost:{}".format(
            epoch+1, n_epoch, cost.item(), test_cost.item()
        ))
    
    plt_epoch.append(epoch+1)
    plt_cost_train.append(cost.item())
    plt_cost_test.append(test_cost.item())


# find the epoch at which the slope of the cost_test graph changes from negative to positive
i = 0
l = 0 # l is the slope between two adjacent epochs.
while True:
    i += 1
    l = (plt_cost_test[i+1]-plt_cost_test[i])/(plt_epoch[i+1]-plt_epoch[i])

    if l > 0:
        print("epoch at which the slope of test cost become positive:", plt_epoch[i])
        print("at that point, test cost :", plt_cost_test[i])
        print("at that point, train cost :", plt_cost_train[i])
        break

# test & accuracy
with torch.no_grad():
    predict = model(test_x)

    correct = (torch.argmax(predict,dim=1) == test_y.squeeze()).sum().item()

    accuracy = 100*correct/len(test_y)

    print("accuracy:", accuracy)

# plot train, test cost - epoch graph
plt.plot(plt_epoch, plt_cost_train, label = 'train')
plt.plot(plt_epoch, plt_cost_test, label = 'test')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend()
plt.show()


print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))