import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train / 127.5 - 1

# Reshape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Generator
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(28*28, activation='tanh'),
    tf.keras.layers.Reshape((28,28,1))
])

# Discriminator
discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile discriminator
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Combine GAN
discriminator.trainable = False

gan = tf.keras.Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Training
epochs = 1000
batch_size = 32

for epoch in range(epochs):

    # Real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]

    # Fake images
    noise = np.random.normal(0,1,(batch_size,100))
    fake_images = generator.predict(noise, verbose=0)

    # Labels
    real_labels = np.ones((batch_size,1))
    fake_labels = np.zeros((batch_size,1))

    # Train discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

    # Train generator
    noise = np.random.normal(0,1,(batch_size,100))
    g_loss = gan.train_on_batch(noise, real_labels)

    # Print progress
    if epoch % 200 == 0:
        print("Epoch:", epoch)

# Generate images
noise = np.random.normal(0,1,(5,100))
generated_images = generator.predict(noise)

# Display images
for i in range(5):
    plt.imshow(generated_images[i].reshape(28,28), cmap='gray')
    plt.show()
