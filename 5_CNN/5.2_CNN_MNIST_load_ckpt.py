# Loading the checkpoint "MNIST_CNN_gygUnig.pt" practice
# Utilize the parameters trained from '4.2_MNIST_CNN.py' to print the accuracy


import numpy as np
import torch
import torch.nn as nn


# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# test data load
test_data = np.loadtxt("../csv_datasets/data_mnist_test.csv", delimiter=',')  # (10000, 785)

# numpy array to torch tensor
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)

# split x, y
test_x = test_data[:, 1:]  # torch.Size([10000, 784])
test_y = test_data[:, :1].long()  # torch.Size([10000, 1])

# x data normalization
test_x = test_x / 255

# x data reshape
test_x = test_x.reshape(10000,1,28,28)


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
    


# test model
test_model = CNN().to(device)
test_model.load_state_dict(torch.load("./checkpoint/MNIST_CNN_gygUnig.pt"))
test_model.eval().to(device)


# accuracy
output = test_model(test_x)
predict = torch.argmax(output, axis=1)

correct = (predict == test_y).sum().item()
accuracy = 100 * correct / len(predict)

print('test accuracy :', accuracy)


