# MNIST - Deep Convolutional Generative Adversarial Networks (DCGAN)
# Use Pytorch
# Use DataLoader


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time

start_time = time.time()
print("Start")



# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# hyper parameters
batch_size = 60
noise_dimension = 100
learning_rate_D = 0.001
learning_rate_G = 0.002
n_epoch = 30




# make Custom DataLoader
class Mnist_DCGAN_Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = np.loadtxt(csv_file, delimiter=',', dtype=np.float32)

        self.train_data = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.n_samples = data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):

        sample = self.train_data[index]
        sample = sample.view(1, 28, 28) # 1D to 3D
        if self.transform:
            sample = self.transform(sample)

        return sample

# transform
transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)),
])

# train dataset
train_dataset = Mnist_DCGAN_Dataset("../csv_file/data_mnist_train.csv", transform=transform)

# Data Loader
train_Loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            # input size : (batch size, 100, 1, 1)

            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size : (batch size, 512, 4, 4)

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size : (batch size, 256, 8, 8)

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size : (batch size, 128, 16, 16)

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size : (batch size, 64, 32, 32)

            nn.ConvTranspose2d(64, 1, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
            # state size : (batch size, 1, 28, 28)

        )

    def forward(self, x):
        out = self.main(x)
        return out


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            # input size : (batch size, 1, 28, 28)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch size, 64, 14, 14)

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch size, 128, 7, 7)

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size : (batch size, 256, 3, 3)

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size : (batch size, 1, 1, 1)
        )

    def forward(self, x):
        return self.main(x)
    

# model
model_G = Generator().to(device)
model_D = Discriminator().to(device)


# cost function - BCE
cost_function = nn.BCELoss()

# optimizer
optimizer_G = optim.Adam(model_G.parameters(), lr = learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(model_D.parameters(), lr = learning_rate_D, betas=(0.5, 0.999))


# # generate random noise - input for Generator
# noise = torch.randn(batch_size, 100, 1, 1).to(device)


# train
G_losses = []
D_losses = []
for epoch in range(n_epoch):
    for i, data in enumerate(train_Loader,0):

        # data into GPU
        data = data.to(device)

        # get the actual size of the batch
        actual_batch_size = data.size(0)

        # generate random noise - input for Generator
        noise = torch.randn(actual_batch_size, 100, 1, 1).to(device)

        # create fake images by passing the random noise to generator
        fake_images = model_G(noise)

        # pass the real images and fake images to the discriminator and get results
        real_output = model_D(data)
        fake_output = model_D(fake_images)

        # real, fake label
        real_label = torch.ones_like(real_output) * 0.9 # Label Smoothing
        fake_label = torch.zeros_like(fake_output)

        # Discriminator's loss function : difference between (real image,1) + difference between (fake image, 0)
        cost_D = cost_function(real_output, real_label) + cost_function(fake_output, fake_label)

        # update the parameters of the discriminator based on cost_D
        optimizer_D.zero_grad()
        cost_D.backward(retain_graph=True)  # generator and discriminator share the same graph 
        optimizer_D.step()

        # Generator's loss function : difference between (fake image, 1)
        cost_G = cost_function(fake_output, torch.ones_like(fake_output))

        # update the parameters of the generator based on cost_G
        optimizer_G.zero_grad()
        cost_G.backward()
        optimizer_G.step()

        # for 100 batch, print Progress
        if (i+1) % 100 == 0:
            print("Epoch {}/{} Batch {}/{} cost_D: {}, cost_G: {}".format(
                epoch+1, n_epoch, i+1, len(train_Loader), cost_D.item(), cost_G.item()
            ))
        G_losses.append(cost_G.item())
        D_losses.append(cost_D.item())


    if (epoch+1) % 1 == 0:
        print("Epoch {}/{}, cost_D: {}, cost_G: {}".format(
            epoch+1, n_epoch, cost_D.item(), cost_G.item()
        ))

# save model
torch.save(model_D.state_dict(), "./checkpoint/7.3_DCGAN_MNIST_2_Pytorch_model_D.pt")
torch.save(model_G.state_dict(), "./checkpoint/7.3_DCGAN_MNIST_2_Pytorch_model_G.pt")




# generate noise for test and create images
test_noise = torch.randn(25, 100, 1, 1).to(device)

# switch the Generator model to evaluation mode
model_G.eval()


# test samples
test_samples = model_G(test_noise)

# scale test samples to [0,255] range
test_samples = (test_samples * 127.5) + 127.5

# convert test samples to image format
test_samples = test_samples.squeeze()


# loss plot
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()



# image plot

# move test_samples to cpu and convert to numpy array
test_samples = test_samples.detach().cpu().numpy()

ims = []
for k in range(25):
    ims.append(Image.fromarray(np.squeeze(test_samples[k,:,:])))

plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(ims[i])
plt.show()



print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))