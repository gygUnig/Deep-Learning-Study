# dog and cat Classification(kaggle data) - CNN practice
# just train version


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import time

start_time = time.time()
print("Start")


# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Size to resize images
image_size = (128, 128)

# Define transformations - resize images and convert to tensors -> to be used when defining the dataset
transform = transforms.Compose([
    transforms.Resize(image_size), # Resize
    transforms.ToTensor() # Convert to Pytorch Tensor
])



# Load data - at this point, torchvision.datasets.ImageFolder finds subdirectories and loads each image,
# classifying them into separate classes. Labels are automatically assigned, like dog as 0, cat as 1, etc.
dataset = datasets.ImageFolder('../image_datasets/dogs_vs_cats_image/train', transform=transform)

# ImageFolder separates each class into directories. The names of the directories are used as class labels and the images in the subdirectories are considered as samples of the respective class.
# When loading images, the given 'transform' is applied to perform operations such as resizing images and converting to tensors.
# ImageFolder automatically labels based on the directory. Class names are sorted in alphabetical order
# In essence, each sample stored in the dataset is a tuple that includes an image tensor (to which transform has been applied) and the label of that image
# ImageFolder objects are easily accessible through indexing. e.g., dataset[0] returns the first image and its label as a tuple


# Print length of the dataset(number of data)
print('number of images: ', len(dataset)) # number of images: 25000

# The shape of each sample stored in the dataset
image,label = dataset[0]
print('image 1 shape: ',image.shape) # image 1 shape:  torch.Size([3, 128, 128])
print('image 1 label: ',label) # image 1 label:  0



# Setting DataLoader
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# DataLoader forms Dataset into a mini-batch form for deep learning model training based on batch
# The entire data in Dataset is sliced by the batch size and provided through DataLoader.
# The defined dataset is put into DataLoader, and DataLoader makes batches with several options (bundling data, shuffling, automatic parallel processing, etc.).
# Batch size and shuffle can be set.




# Design CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.fc1 = nn.Linear(16*16*128, 625, bias=True)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU()
        )

        self.fc2 = nn.Linear(625,2,bias=True)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self,x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out
    


# Define CNN model
model = CNN().to(device)

# hyper parameters
learning_rate = 0.001
n_epochs = 15

# cost function, optimizer
cost_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# train
for epoch in range(n_epochs):

    for X, Y in data_loader: # X : image data, Y : label

        X = X.to(device)
        Y = Y.to(device)

        # hypothesis function
        hypothesis = model(X)

        # cost
        cost = cost_function(hypothesis, Y)

        optimizer.zero_grad()  # reset gradiants
        cost.backward()  # backpropagate errors
        optimizer.step()   # update model parameters

    print('epoch {}/{}, train_cost: {}'.format(
        epoch+1, n_epochs, cost.item()
    ))

# save model
torch.save(model.state_dict(),"./checkpoint/5.3_CNN_dog_cat_just_train_gygUnig.pt")



print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))