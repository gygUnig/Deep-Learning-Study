# MNIST Classification practice : Recurrent Neural Network(RNN)
# Using Pytorch


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import time

start_time = time.time()
print("Start")


# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
n_epoch = 10
learning_rate = 0.001


# make Custom DataLoader
class MnistDataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)

        self.x_data = torch.tensor(data[:, 1:], dtype=torch.float32).to(device)
        self.y_data = torch.tensor(data[:, :1], dtype=torch.float32).to(device)
        self.n_samples = data.shape[0]  # 60000

    # Return the total length of the dataset
    def __len__(self):
        return self.n_samples
    
    # Return the input-output data corresponding to the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


# train, test dataset
train_dataset = MnistDataset('../csv_datasets/data_mnist_train.csv')
test_dataset = MnistDataset('../csv_datasets/data_mnist_test.csv')


# Data Loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)


# RNN - LSTM model
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0)) # out : tensor of shape (batch_size, seq_length, hidden_size_output)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
    
# model
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# cost function, optimizer
cost_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# train
for epoch in range(n_epoch):

    for X, Y in train_loader:

        X = X.to(device)
        Y = Y.squeeze(dim = -1)
        Y = Y.type(torch.LongTensor)
        Y = Y.to(device)

        X = X.reshape(-1, sequence_length, input_size).to(device)

        # hypothesis
        hypo = model(X)

        # cost
        cost = cost_function(hypo, Y)

        optimizer.zero_grad()  # reset gradiants
        cost.backward()  # backpropagate errors
        optimizer.step()  # updata model parameters

    print('epoch {}/{} train_cost:{}'.format(
        epoch+1, n_epoch, cost.item()
    ))

# save model
torch.save(model.state_dict(), "./checkpoint/6.1_LSTM_MNIST_Pytorch.pt")


# test & accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X, Y in test_loader:

        X = X.to(device)
        Y = Y.squeeze(dim = -1)
        Y = Y.type(torch.LongTensor)
        Y = Y.to(device)

        X = X.reshape(-1, sequence_length, input_size).to(device)

        output = model(X)
        predict = torch.argmax(output.data, axis=1)
        total += Y.size(0)

        correct += (predict == Y).sum().item()

accuracy = 100 * correct / total
print('Test Accuracy : {} %'.format(accuracy))


print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))


