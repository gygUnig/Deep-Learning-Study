# MNIST - Generative Adversarial Networks (GAN)
# Using tensorflow2

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

start_time = time.time()
print("====Start====")

# Data Load
train_data = np.loadtxt("../csv_datasets/data_mnist_train.csv", delimiter=',') # (60000, 785)

print("====Data Loaded====")

# Data normalization -> [-1, 1]
train_data = (train_data[:, 1:] - 127.5) / 127.5


# hyper parameters
learning_rate_D = 0.001
learning_rate_G = 0.002
n_epoch = 300
batch_size = 3000


# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.fc = models.Sequential([
            layers.Dense(512, activation=layers.LeakyReLU(alpha=0.2)),
            layers.Dense(256, activation=layers.LeakyReLU(alpha=0.2)),
            layers.Dense(128, activation=layers.LeakyReLU(alpha=0.2)),
            layers.Dense(1, activation='sigmoid')
        ])
        
    def call(self, x):
        return self.fc(x)
    
    
# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc = models.Sequential([
            layers.Dense(128, activation='relu'), # tensorflow는 모델이 동적 입력크기를 받을 수 있으므로 입력 크기를 정의하지 않아도 무방
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(1*28*28, activation='tanh')
        ])
    
    def call(self, x):
        return self.fc(x)
    
# Discriminator, Generator model
model_D = Discriminator()
model_G = Generator()

# cost function
cost_function = tf.keras.losses.BinaryCrossentropy()

# optimizer
optimizer_D = optimizers.Adam(learning_rate=learning_rate_D, beta_1=0.5)
optimizer_G = optimizers.Adam(learning_rate=learning_rate_G, beta_1=0.5)

print("====train start====")

# train
G_losses=[]
D_losses=[]
for epoch in tqdm(range(n_epoch)):
    
    for b in range(int(len(train_data)/batch_size)):
        
        # generate random noise - input for Generator
        noise = tf.random.normal([batch_size, 100])
        
        # real images
        real_images = train_data[batch_size*b : batch_size*(b+1), :]
        
        # persistant=True인 경우, 동일한 계산에 대해 여러 번 gradient를 계산할 수 있게 해준다.
        with tf.GradientTape(persistent=True) as tape:
            
            # create fake images by passing the random noise to generator
            fake_images = model_G(noise)
            
            # pass the real images and fake images to the discriminator and get results
            real_output = model_D(real_images)
            fake_output = model_D(fake_images)
            
            # real, fake label
            real_label = tf.ones_like(real_output)
            fake_label = tf.zeros_like(fake_output)
            
            
            # Discriminator's cost function
            cost_D = cost_function(real_label, real_output) + cost_function(fake_label, fake_output)
            
            # Generator's cost function
            cost_G = cost_function(tf.ones_like(fake_output), fake_output)
            
        
        # gradient 계산
        # tape.gradient : 주어진 비용 함수에 대한 변수들의 gradient 계산
        # 그 결과로 grads_D 는 Discriminator의 각 변수에 대한 gradients 값을 포함하는 리스트이다.
        grads_D = tape.gradient(cost_D, model_D.trainable_variables)
        grads_G = tape.gradient(cost_G, model_G.trainable_variables)
        
        # Gradient 적용 및 parameter Update
        optimizer_D.apply_gradients(zip(grads_D, model_D.trainable_variables))
        optimizer_G.apply_gradients(zip(grads_G, model_G.trainable_variables))
        
        G_losses.append(cost_G.numpy())
        D_losses.append(cost_D.numpy())
        
    if (epoch+1) % 10 == 0:
        print("Epoch:{}/{} cost_D:{}, cost_G:{}".format(
            epoch + 1, n_epoch, cost_D.numpy(), cost_G.numpy()
        ))


# model save
model_G.save('./checkpoint/7.6_GAN_MNIST_Tensorflow_model_G.tf')
model_D.save('./checkpoint/7.6_GAN_MNIST_Tensorflow_model_D.tf')


# Generate noise for testing and create images
# Number of images to create: 25, dimension matching the input of the generator: 100
test_noise = tf.random.normal([25, 100])

# Test samples - pass through Tanh activation function in model_G so the range of elements is [-1,1]
test_samples = model_G(test_noise)

# Scale test samples to [0,255] range
test_samples = (test_samples * 127.5) + 127.5

# Convert test samples to image format
test_samples = tf.reshape(test_samples, (25, 28, 28, 1))

# Loss plot
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Image plot
# Convert test samples to numpy array
test_samples = test_samples.numpy().squeeze()

plt.figure(figsize=(5, 5))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_samples[i], cmap='gray')
    plt.axis('off') # Optionally, you can turn off the axis

plt.show()           
        
print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)


print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))        
            
    






