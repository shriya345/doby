# LAB 5 - Transfer Learning using MobileNet (Simple Version)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import fashion_mnist
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert grayscale to RGB and resize to 32x32
x_train = np.stack((x_train,)*3, axis=-1)
x_test = np.stack((x_test,)*3, axis=-1)

x_train = tf.image.resize(x_train, (32,32))
x_test = tf.image.resize(x_test, (32,32))

# Load pretrained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(32,32,3))

# Freeze pretrained layers
base_model.trainable = False

# Build transfer learning model
model = Sequential([
    base_model,
    Flatten(),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=3, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", accuracy)
