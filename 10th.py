import tensorflow as tf
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=3)

# Evaluate baseline model
loss, accuracy = model.evaluate(x_test, y_test)

print("Baseline Accuracy:", accuracy)

# Simple pruning
weights = model.get_weights()

for i in range(len(weights)):
    weights[i][np.abs(weights[i]) < 0.1] = 0

model.set_weights(weights)

# Evaluate pruned model
loss, pruned_accuracy = model.evaluate(x_test, y_test)

print("Pruned Accuracy:", pruned_accuracy)

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_model = converter.convert()

print("Quantized Model Size:",
      len(quantized_model)/1024, "KB")
