# MNIST Classification practice : Fully Connected Neural Network
# Using tensorflow2


import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm


start_time = time.time()
print("=========start========")

# train, test data load
data = np.loadtxt("../Datasets/mnist_train.csv", delimiter=',', dtype=np.float32, skiprows=1)  # (60000, 785)
test_data = np.loadtxt("../Datasets/mnist_test.csv", delimiter=',', dtype=np.float32, skiprows=1)  # (10000, 785)

print("=========data loaded========")


# data shuffle
np.random.seed(1)
np.random.shuffle(data)
np.random.shuffle(test_data)

# split x,y
train_x = data[:, 1:] 
train_y = data[:, :1].astype(int)

test_x = test_data[:, 1:]  
test_y = test_data[:, :1].astype(int)


# x - normalization (mean max scale)
train_x = train_x / 255
test_x = test_x / 255


# model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10)
])

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

# cost function
# tf.keras.losses.SparseCategoricalCrossentropy 는 클래스 레이블이 정수로 제공되는 다중 클래스 분류 문제에 사용되는 손실 함수
# logit이 제공되면 from_logits=True로 설정하고, 확률이 제공되면 from_logits=False로 설정
cost_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# tf.keras.losses.CategoricalCrossentropy 와의 차이 : 
# CategoricalCrossentropy 는 클래스 레이블이 원-핫 인코딩 형태로 제공되어야 함
# SparseCategoricalCrossentropy 는 클래스 레이블이 정수 형태로 제공되어야 함


# batch size
batch_size = 3000

# number of epoch
n_epoch = 1000



print("=========train start========")

plt_epoch = []
plt_cost_train = []
plt_cost_test = []
# train
for epoch in tqdm(range(n_epoch)):
    
    for b in range(int(len(train_y)/batch_size)):
        
        with tf.GradientTape() as tape:
            
            # cost
            logits = model(train_x[b * batch_size : (b+1) * batch_size, :])
            cost = cost_function(train_y[b * batch_size : (b+1) * batch_size], logits)
            
        gradients = tape.gradient(cost, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    test_cost = cost_function(test_y, model(test_x))
    
    if (epoch + 1) % 100 == 0:
        print("epoch {}/{} train_cost:{} test_cost:{}".format(
            epoch+1, n_epoch, cost.numpy(), test_cost.numpy()
        ))    
        
    
    plt_epoch.append(epoch+1)
    plt_cost_train.append(cost.numpy())
    plt_cost_test.append(test_cost.numpy())
    
# find the epoch at which the slope of the cost_test graph changes from negative to positive
i = 0
l = 0 # l is the slope between two adjacent epochs.
while True:
    i += 1
    l = (plt_cost_test[i+1]-plt_cost_test[i])/(plt_epoch[i+1]-plt_epoch[i])
    
    if l > 0:
        print("epoch at which the slope of test cost become positive:", plt_epoch[i])
        print("at that point, test cost :", plt_cost_test[i])
        print("at that point, train cost :", plt_cost_train[i])
        break

print("=========train End========")


# test & accuracy
predict = model(test_x)
correct = np.sum(np.argmax(predict, axis=1) == test_y.squeeze())
accuracy = 100 * correct / len(test_y)

print("accuracy :", accuracy)


# plot trian, test cost - epoch graph
plt.plot(plt_epoch, plt_cost_train, label='train')
plt.plot(plt_epoch, plt_cost_test, label='test')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend()
plt.show()



# running time
print("End")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print("{}:{}:{:.2f}".format(int(hours), int(minutes), seconds))






