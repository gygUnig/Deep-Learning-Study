# Logistic Regression Practice : score & pass/fail
# Using Pytorch

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# data
train_x = torch.FloatTensor([[15],[24],[57],[78],[90],[114]]) # score
train_y = torch.FloatTensor([[0],[0],[0],[1],[1],[1]]) # pf

# init W, b
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W,b], lr = 0.005)

# number of epoch
n_epoch = 30000

# train
for epoch in range(n_epoch):

    # hypothesis function
    hypo = torch.sigmoid(train_x * W + b)

    # cost function
    cost = F.binary_cross_entropy(hypo, train_y)

    optimizer.zero_grad()  # reset gradiants
    cost.backward()  # backpropagate errors
    optimizer.step()  # update model parameters

    if (epoch + 1) % 100 == 0:
        print("epoch {}/{}, train_cost:{}".format(
            epoch+1, n_epoch, cost.item()
        ))


# predict & real
predict = torch.round(hypo).squeeze().detach()
real = train_y.squeeze()

print('predict :', predict)
print('real    :', real)


# plot
x_plt = np.arange(0,120,0.5)
y_plt = 1/(1+np.exp(-(x_plt*(W.detach().numpy()) + b.detach().numpy())))

plt.plot(x_plt, y_plt)
plt.scatter(train_x.numpy(), train_y.numpy())
plt.show()
