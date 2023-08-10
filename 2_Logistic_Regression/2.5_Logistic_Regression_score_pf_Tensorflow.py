# Logistic Regression Practice : score & pass/fail
# Using tensorflow2


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# data
train_x = np.array([[15],[24],[57],[78],[90],[114]]) # score
train_y = np.array([[0],[0],[0],[1],[1],[1]]) # pf

# init W, b
W = tf.Variable(tf.zeros(1,), dtype=tf.float32)
b = tf.Variable(tf.zeros(1,), dtype=tf.float32)

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.005)

# number of epoch
n_epoch = 30000


# trian
for epoch in tqdm(range(n_epoch)):
    
    with tf.GradientTape() as tape:
        
        # hypothesis function
        hypo = tf.sigmoid(train_x * W + b)
        
        # cost function
        BCE = tf.keras.losses.BinaryCrossentropy()
        cost = BCE(train_y, hypo)
        
    # gradients
    gradients = tape.gradient(cost, [W, b])
    
    # Update model parameters
    optimizer.apply_gradients(zip(gradients, [W, b]))
    

    if (epoch + 1) % 100 == 0:
        print("epoch {}/{}, train_cost:{}".format(
            epoch+1, n_epoch, cost.numpy()
        ))    

# predict & real
predict = tf.round(hypo).numpy().squeeze()
real = train_y.squeeze()

print("predict :",predict)
print("real    :", real)
    
    
# plot 
x_plt = np.arange(0, 120, 0.5)
y_plt = 1 / (1 + np.exp(-(x_plt * W.numpy() + b.numpy())))

plt.plot(x_plt, y_plt)
plt.scatter(train_x, train_y)
plt.show()



