import tensorflow as tf
import numpy as np

# Load dataset
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0

# Split training and validation
X_train = X_train_full[:-5000]
X_val = X_train_full[-5000:]

y_train = y_train_full[:-5000]
y_val = y_train_full[-5000:]

# Add channel dimension
X_train = np.array(X_train)[..., np.newaxis]
X_val = np.array(X_val)[..., np.newaxis]
X_test = np.array(X_test)[..., np.newaxis]

# CNN Model Function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Adam Optimizer
adam_model = create_model()

adam_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with Adam")
adam_model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

adam_loss, adam_acc = adam_model.evaluate(X_test, y_test)

# RMSProp Optimizer
rms_model = create_model()

rms_model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training with RMSProp")
rms_model.fit(X_train, y_train, epochs=3, validation_data=(X_val, y_val))

rms_loss, rms_acc = rms_model.evaluate(X_test, y_test)

# Compare Accuracy
print("Adam Accuracy:", adam_acc)
print("RMSProp Accuracy:", rms_acc)
