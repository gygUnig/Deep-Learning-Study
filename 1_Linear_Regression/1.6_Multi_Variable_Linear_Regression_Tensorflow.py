# Multi variable Linear Regression practice : test socre
# using tensorflow2


import tensorflow as tf
import numpy as np


# data load
data = np.loadtxt('../csv_datasets/data_test_score.csv', delimiter=',', dtype=np.float32)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# split train data, test data
train_x = data[:20, :3]
train_y = data[:20, 3:]

test_x = data[20:, :3]
test_y = data[20:, 3:]


# init W, b
W = tf.Variable(tf.zeros((3,1)), dtype=tf.float32)
b = tf.Variable(tf.zeros((1,)), dtype=tf.float32)
# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

# number of epoch
n_epoch = 10000

# train
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        
        # hypothesis function
        hypo = tf.matmul(train_x, W) + b
        
        # cost function
        cost = tf.reduce_mean(tf.square(hypo - train_y))
        
    # gradients
    gradients = tape.gradient(cost, [W, b])
    
    # update model parameters
    optimizer.apply_gradients(zip(gradients, [W,b]))
    
    if (epoch+1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost.numpy()
        ))
        
# test
predict = tf.matmul(test_x, W) + b
print("Predict :", predict.numpy().squeeze())
print("real    :", test_y.squeeze())
        