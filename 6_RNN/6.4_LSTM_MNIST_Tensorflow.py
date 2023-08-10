# MNIST Classification practice : Recurrent Neural Network(RNN)
# Using tensorflow2

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
import time


start_time = time.time()
print("====Start====")


# hyperparameters
sequence_length = 28  # 시퀀스의 길이 - MNIST 이미지의 한 줄을 나타낸다.
input_size = 28  # 입력 벡터의 크기 - MNIST 이미지의 한 행을 나타낸다.
hidden_size = 128 
num_classes = 10
batch_size = 100
n_epochs = 10
learning_rate = 0.001


# load dataset
train_data = np.loadtxt('../csv_datasets/data_mnist_train.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('../csv_datasets/data_mnist_test.csv', delimiter=',', dtype=np.float32)


print("====data loaded====")


# split dataset
train_x = train_data[:, 1:]
train_y = train_data[:, :1].astype(int)

test_x = test_data[:, 1:]
test_y = test_data[:, :1].astype(int)


# Reshape
train_x = train_x.reshape(-1,sequence_length, input_size)
test_x = test_x.reshape(-1, sequence_length, input_size)



# LSTM model
model = models.Sequential([
    layers.LSTM(hidden_size, return_sequences=True, input_shape = (sequence_length, input_size)),
    # return_sequences=True : 이 층이 시퀀스의 전체 출력을 반환하도록 지시하는 것. (기본값 : False)
    # 이렇게 하면 다음 LSTM 층이 시퀀스의 전체 출력을 입력으로 받을 수 있다.
    layers.LSTM(hidden_size),
    layers.Dense(num_classes, activation='softmax')
])


# compile model
model.compile(
    optimizer = optimizers.Adam(learning_rate=learning_rate),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("====train start====")


# train 
model.fit(train_x, train_y, batch_size=batch_size, epochs=n_epochs)

# evaluate model
test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
print("Test accuracy : {} %".format(test_accuracy * 100))



print("====End====")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print("{}:{}:{:.2f}".format(
    int(hours), int(minutes), seconds
))
