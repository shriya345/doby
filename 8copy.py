import numpy as np
import tensorflow as tf

# Sample text
text = "deep learning is fun and lstm learns sequences"

# Unique characters
chars = sorted(list(set(text)))

# Character mapping
char_to_idx = {c:i for i,c in enumerate(chars)}
idx_to_char = {i:c for i,c in enumerate(chars)}

# Prepare sequences
seq_length = 5

X = []
Y = []

for i in range(len(text) - seq_length):
    seq = text[i:i+seq_length]
    next_char = text[i+seq_length]

    X.append([char_to_idx[ch] for ch in seq])
    Y.append(char_to_idx[next_char])

X = np.array(X)
Y = np.array(Y)

# Reshape input
X = X.reshape((X.shape[0], X.shape[1], 1))
X = X / len(chars)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seq_length,1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, Y, epochs=50)

# Test prediction
test = "deep "

x_input = np.array([[char_to_idx[ch] for ch in test]])
x_input = x_input.reshape((1, seq_length, 1))
x_input = x_input / len(chars)

prediction = model.predict(x_input)

predicted_char = idx_to_char[np.argmax(prediction)]

print("Input:", test)
print("Predicted Character:", predicted_char)
