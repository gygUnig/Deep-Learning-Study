# Linear Regression Practice
# Using tensorflow2

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# data
train_x = np.array([[100],[125],[150],[190],[206]]) # weight
train_y = np.array([[105],[122],[155],[176],[207]]) # height


# init W, b
# tf.Variable은 변수 텐서를 생성하는 것. 
# 학습 중에 변경될 수 있는 텐서로, 가중치나 편향과 같은 모델 파라미터에 주로 사용 됨
W = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)
b = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)


# number of epochs
n_epoch = 10000

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.00001)


# Train
for epoch in range(n_epoch):
    
    # Tensorflow에서는 tf.GradientTape를 사용하여 Gradient를 자동으로 계산한다.
    # 이 구문은 해당 컨텍스트 안에서 수행된 연산을 기록하여 나중에 미분할 수 있게 한다.
    with tf.GradientTape() as tape:
        
        # Hypothesis function
        hypo = W * train_x + b
        
        # Cost function
        cost = tf.reduce_mean(tf.square(hypo - train_y))
        
    
    # Gradients
    gradients = tape.gradient(cost, [W,b]) # 기록된 연산을 바탕으로 비용 함수에 대한 [W, b]의 Gradient를 계산한다.
    
    # Update model parameters
    optimizer.apply_gradients(zip(gradients, [W,b]))
    
    
    if (epoch+1) % 100 == 0:
        print("epoch:{}/{} w:{} b:{} cost:{}".format(
            epoch+1, n_epoch, W.numpy()[0], b.numpy()[0], cost.numpy()
        ))
        
# Plot
x_plt = np.arange(0,300,0.5)
y_plt = W.numpy() * x_plt + b.numpy()
train_x_plt = train_x
train_y_plt = train_y

plt.plot(x_plt, y_plt)
plt.scatter(train_x_plt, train_y_plt)
plt.xlabel('weight')
plt.ylabel('height')
plt.show()