# Logistic Regression Practice : diabetes
# Using tensorflow2

import tensorflow as tf
import numpy as np
from tqdm import tqdm


# load data
data = np.loadtxt("../Datasets/data_diabetes.csv", delimiter=',', dtype=np.float32)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# split train data, test data
train_x = data[:600, :8]
train_y = data[:600, 8:]
test_x = data[600:, :8]
test_y = data[600:, 8:]


# init W, b
W = tf.Variable(tf.zeros((8, 1)), dtype=tf.float32)
b = tf.Variable(tf.zeros((1, )), dtype=tf.float32)

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)

# number of epoch
n_epoch = 10000


# train
for epoch in tqdm(range(n_epoch)):
    
    with tf.GradientTape() as tape:
        
        # hypothesis function
        hypo = tf.sigmoid(tf.matmul(train_x, W)+b)
        
        # cost function
        BCE = tf.keras.losses.BinaryCrossentropy()
        cost = BCE(train_y, hypo)
    
    # gradients
    gradients = tape.gradient(cost, [W, b])
    
    # model parameters update
    optimizer.apply_gradients(zip(gradients, [W, b]))
    

    if (epoch + 1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch+1, n_epoch, cost.numpy()
        ))
        
# test & accuracy
predict = tf.sigmoid(tf.matmul(test_x, W)+b)
predict = tf.round(predict)
        
# tf.cast(predict == test_y, dtype=tf.float32)는 True를 1.0으로, False를 0.0으로 변환한다.
correct = tf.reduce_sum(tf.cast(predict == test_y, dtype=tf.float32)).numpy()

print("test accuracy :", (correct/len(test_y))*100)
        






