# load checkpoint of 5.4_dog_cat_DCGAN


import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt



# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size : (batch size, 32, 64, 64)


            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size : (batch size, 3, 128, 128)
        )

    def forward(self, x):
        out = self.main(x)
        return out




# Generator model
model_G = Generator().to(device)
model_G.load_state_dict(torch.load("./checkpoint/7.4_DCGAN_dogs_vs_cats_model_G_v1.pt"))
model_G.eval()


# test noise
test_noise = torch.randn(25, 100, 1, 1).to(device)

# test samples
test_samples = model_G(test_noise)
test_samples = test_samples.detach().cpu().numpy()

# image plot
ims = []
for k in range(25):
    
    # test samples : [-1, 1] range -> [0, 1]
    img = (test_samples[k,:,:,:] * 0.5) + 0.5

    # Transpose from (3, 128, 128) to (128, 128, 3)
    img = np.transpose(img, (1,2,0))

    # [0, 1] -> [0, 255]
    img = (img*255).astype(np.uint8)
    ims.append(Image.fromarray(img))
    

plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(ims[i])
plt.show()
