# Softmax Regression Practice : Zoo
# Using tensorflow2

import tensorflow as tf
import numpy as np
from tqdm import tqdm

# load data
data = np.loadtxt('../Datasets/data_zoo.csv',delimiter=',',dtype=np.float32)

# data shuffle
np.random.seed(1)
np.random.shuffle(data)

# split train data, test data
train_x = data[:80, :16]
train_y = data[:80, 16:].astype(int)
test_x = data[80:, :16]
test_y = data[80:, 16:].astype(int)

# one hot encoding
train_y_one_hot = tf.one_hot(train_y.squeeze(), depth=7)

# init W, b
W = tf.Variable(tf.zeros((16,7)), dtype=tf.float32)
b = tf.Variable(tf.zeros((1,7)), dtype=tf.float32)

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# number of epoch
n_epoch = 10000


# train

for epoch in tqdm(range(n_epoch)):
    
    with tf.GradientTape() as tape:
        
        # hypothesis function
        # logits는 모델의 출력을 나타내기 전에 활성화 함수(시그모이드나 소프트맥스)를 거치지 않은 원시 출력값을 의미한다
        # 즉, 확률로 해석하기 전의 원시 값
        logits = tf.matmul(train_x, W) + b
        hypo = tf.nn.softmax(logits, axis=1)
        
        # cost function
        cost = -tf.reduce_mean(tf.reduce_sum(train_y_one_hot * tf.math.log(hypo), axis=1))
        
    # gradients
    gradients = tape.gradient(cost, [W, b])
    
    # model parameters update
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    
    if (epoch + 1) % 100 == 0:
        print("epoch:{}/{} train_cost:{}".format(
            epoch + 1, n_epoch, cost.numpy()
        ))        
        
# test & accuracy
test_logits = tf.matmul(test_x, W) + b
predict = tf.argmax(tf.nn.softmax(test_logits, axis=1), axis=1)
real = test_y.squeeze()

correct = tf.reduce_sum(tf.cast(predict == real, dtype=tf.float32)).numpy()

print('test correct/entire = {}/{},'.format(correct, len(real)), 'accuracy :', 100 * correct / len(real))