import numpy as np
import tensorflow as tf

# Create sequence data
x = np.linspace(0, 50, 100)
y = np.sin(x)

# Prepare training data
X = []
Y = []

seq_length = 10

for i in range(len(y) - seq_length):
    X.append(y[i:i+seq_length])
    Y.append(y[i+seq_length])

X = np.array(X)
Y = np.array(Y)

# Reshape for RNN
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(10, activation='relu', input_shape=(seq_length,1)),
    tf.keras.layers.Dense(1)   # ONLY 1 output
])

# Compile
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(X, Y, epochs=50)

# Predict
predictions = model.predict(X)

# Output
print("Expected:", Y[:5])
print("Predicted:", predictions[:5].flatten())
