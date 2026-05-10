import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train, _), (X_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add noise
noise = 0.3

X_train_noisy = X_train + noise * np.random.randn(*X_train.shape)
X_test_noisy = X_test + noise * np.random.randn(*X_test.shape)

# Keep values between 0 and 1
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# Add channel dimension
X_train = np.array(X_train)[..., np.newaxis]
X_test = np.array(X_test)[..., np.newaxis]

X_train_noisy = np.array(X_train_noisy)[..., np.newaxis]
X_test_noisy = np.array(X_test_noisy)[..., np.newaxis]

# Autoencoder Model
model = tf.keras.Sequential([

    # Encoder
    tf.keras.layers.Conv2D(
        16, 3,
        activation='relu',
        padding='same',
        input_shape=(28,28,1)
    ),

    tf.keras.layers.MaxPooling2D(2),

    # Decoder
    tf.keras.layers.Conv2DTranspose(
        16, 3,
        strides=2,
        activation='relu',
        padding='same'
    ),

    tf.keras.layers.Conv2D(
        1, 3,
        activation='sigmoid',
        padding='same'
    )
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train
model.fit(
    X_train_noisy,
    X_train,
    epochs=5,
    validation_data=(X_test_noisy, X_test)
)

# Predict cleaned images
output = model.predict(X_test_noisy[:5])

# Display noisy image
plt.imshow(X_test_noisy[0].reshape(28,28), cmap='gray')
plt.title("Noisy Image")
plt.show()

# Display denoised image
plt.imshow(output[0].reshape(28,28), cmap='gray')
plt.title("Denoised Image")
plt.show()
