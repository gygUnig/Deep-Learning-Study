# MNIST Classification practice : Convolutional Neural Network
# Using Pytorch


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time

start_time = time.time()
print("Start")

# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# train, test data load
train_data = np.loadtxt("../csv_datasets/data_mnist_train.csv", delimiter=',')  # (60000, 785)
test_data = np.loadtxt("../csv_datasets/data_mnist_test.csv", delimiter=',')  # (10000, 785)


# data shuffle
np.random.seed(1)
np.random.shuffle(train_data)
np.random.shuffle(test_data)


# numpy array to torch tensor
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)


# split x,y
train_x = train_data[:, 1:]  # torch.Size([60000, 784])
train_y = train_data[:, :1].long()  # torch.Size([60000, 1])

test_x = test_data[:, 1:]  # torch.Size([10000, 784])
test_y = test_data[:, :1].long()  # torch.Size([10000, 1])


# x data normalization (mean max scale) : X = (x - x_min)/(x_max - x_min)
# MNIST data : 0 ~ 255
train_x = train_x / 255
test_x = test_x / 255


# x data reshape
# data needs to be in a 28x28 format rather than 784 to feed into the CNN
# grayscale image -> only one channel
# Pytorch accepts input tensors in the shape of (batch size, channels, height, width)
train_x = train_x.reshape(60000,1,28,28)
test_x = test_x.reshape(10000,1,28,28)


# hyper parameters
learning_rate = 0.001
n_epoch = 15
batch_size = 3000


# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.keep_prob = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # after layer 1 : (batch_size, 32, 14, 14)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # after layer 2 : (batch_size, 64, 7, 7)

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        # after layer 3 : (batch_size, 128, 4, 4)

        self.fc1 = nn.Linear(4*4*128, 625, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        self.layer4 = nn. Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=1-self.keep_prob)
        )
        # after layer 4 : (batch_size, 625)


        self.fc2 = nn.Linear(625, 10, bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)
        # after fc2 : (batch_size, 10)


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out


# CNN model
model = CNN().to(device)

# cost function, optimizer
cost_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# train
for epoch in range(n_epoch):

    for b in range(int(len(train_y)/batch_size)):

        # cost
        cost = cost_function(model(train_x[batch_size*b:batch_size*(b+1),:,:,:]), train_y[batch_size*b:batch_size*(b+1),:].squeeze())

        optimizer.zero_grad()  # reset gradiants
        cost.backward()  # backpropagate errors
        optimizer.step()  # update model parameters

    print('epoch {}/{} train_cost:{}'.format(
        epoch+1, n_epoch, cost.item()
    ))


# save model
torch.save(model.state_dict(), "./checkpoint/5.1_CNN_MNIST_Pytorch.pt")


# test & accuracy
predict = model(test_x.to(device))
predict = torch.argmax(predict, axis = 1)

correct = (predict == test_y.squeeze()).sum().item()
print('test accuracy : ',100*correct/len(predict))


print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))