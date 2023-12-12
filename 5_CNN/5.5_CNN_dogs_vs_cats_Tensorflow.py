# dog and cat Classification(kaggle data) - CNN practice
# Using tensorflow2

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

start_time = time.time()
print("====Start====")


# Size to resize images
image_size = (128, 128)

# ImageDataGenerator를 사용해 이미지 데이터의 전처리를 설정한다.
# validation_split = 0.2 : 전체 데이터의 20%는 검증 데이터로 분리한다
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# Load Train Data
# train_data는 훈련 데이터셋의 이미지와 레이블을 모두 포함한다.
train_data = train_datagen.flow_from_directory(
    '../Datasets/dogs_vs_cats_image/train',
    target_size=image_size,
    batch_size=64,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# Load Validation Data
val_data = train_datagen.flow_from_directory(
    '../Datasets/dogs_vs_cats_image/train',
    target_size=image_size,
    batch_size=64,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

print("====data loaded====")


# hyper parameters
learning_rate = 0.001
n_epochs = 15
batch_size = 64



# model
model = models.Sequential([
    layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', input_shape=(128,128,3), padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(625, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# model Compile
model.compile(
    optimizer = optimizers.Adam(learning_rate=learning_rate),
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

print("====train start====")


# Train
model.fit(train_data, validation_data=val_data, epochs=n_epochs)

# Evaluate on Validation Data
loss, accuracy = model.evaluate(val_data)
print("Validation accuracy : {} %".format(accuracy*100))



print("====End====")
end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"{int(hours)}:{int(minutes)}:{seconds:.2f}")

