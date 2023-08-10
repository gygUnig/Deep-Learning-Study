# MNIST Classification practice : Convolutional Neural Network
# Using tensorflow2


import tensorflow as tf
import numpy as np
import time


start_time = time.time()
print("===========Start============")


# Load data
train_data = np.loadtxt("../csv_datasets/data_mnist_train.csv", delimiter=',')
test_data = np.loadtxt("../csv_datasets/data_mnist_test.csv", delimiter=',')

print("========data loaded==========")

# Shuffle data
np.random.seed(1)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

# Split data
train_x = train_data[:, 1:]
train_y = train_data[:, :1].astype(int)

test_x = test_data[:, 1:]
test_y = test_data[:, :1].astype(int)

# data Normalize
train_x = train_x / 255
test_x = test_x / 255


# data reshape
# tensorflow의 경우 데이터가 (배치 크기, 높이, 너비, 채널 수) 형태로 구성된다.
train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)

# hyper parameters
learning_rate = 0.001
n_epoch = 15
batch_size = 64


# model
# tensorflow 에서는 padding 인자에 same 또는 valid를 사용할 수 있다.
# same은 출력 크기가 입력과 동일하게 유지되도록 패딩을 자동으로 조정하고, valid는 패딩을 적용하지 않는다.
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    
    tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(625, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compile
# 텐서플로우에서 모델을 학습하기 전에 필요한 몇 가지 구성을 설정하는 단계
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'] # metrics : 모델의 성능을 평가하는 데 사용되는 지표를 리스트 형태로 설정. ['accuracy']는 정확도를 평가 지표로 사용하겠다는 것
)

print("========train start==========")

# train
# verbose는 학습 중 출력할 정보의 양을 제어한다
# 0은 아무것도 출력하지 않고, 1은 진행률 막대를 포함한 중간 출력을, 2는 에포크당 한 줄의 로그를 출력한다.
model.fit(train_x, train_y, epochs=n_epoch, batch_size=batch_size, verbose=1)


# model save
model.save('./checkpoint/5.4_CNN_MNIST_Tensorflow.h5')


# evaluation
test_loss, test_accuracy = model.evaluate(test_x, test_y)
print("Test accuracy : ", test_accuracy)