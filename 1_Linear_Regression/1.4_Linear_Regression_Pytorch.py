# Linear Regression Practice
# Using Pytorch

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# data
train_x = torch.FloatTensor([[100],[125],[150],[190],[206]]) # weight
train_y = torch.FloatTensor([[105],[122],[155],[176],[207]]) # height


# init W, b
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = optim.SGD([W,b], lr = 0.00001)

# number of epoch
n_epoch = 10000

# train
for epoch in range(n_epoch):

    # hypothesis function
    hypo = W * train_x + b

    # cost function
    cost = torch.mean((hypo - train_y) ** 2)

    optimizer.zero_grad()  # reset gradients
    cost.backward()  # backpropagate errors
    optimizer.step()  # update model parameters

    if (epoch+1) % 100 == 0:
        print("epoch:{}/{} w:{} b:{} cost:{}".format(
            epoch+1, n_epoch, W.item(), b.item(),cost.item() 
        ))

# plot
x_plt = np.arange(0,300,0.5)
y_plt = W.detach() * x_plt + b.detach()
train_x_plt = train_x.numpy()
train_y_plt = train_y.numpy()

plt.plot(x_plt, y_plt)
plt.scatter(train_x_plt, train_y_plt)
plt.xlabel('weight')
plt.ylabel('height')
plt.show()

# epoch:10000/10000 w:0.9866237044334412 b:0.14951153099536896 cost:47.048526763916016