import numpy as np
import tensorflow as tf

# Sample text
text = "hello deep learning"

# Unique characters
chars = sorted(set(text))

# Dictionaries
char_to_idx = {}
idx_to_char = {}

for i, ch in enumerate(chars):
    char_to_idx[ch] = i
    idx_to_char[i] = ch

# Prepare data
seq_length = 5

X = []
Y = []

for i in range(len(text) - seq_length):

    seq = text[i:i+seq_length]
    next_char = text[i+seq_length]

    temp = []

    for ch in seq:
        temp.append(char_to_idx[ch])

    X.append(temp)
    Y.append(char_to_idx[next_char])

# Convert to arrays
X = np.array(X)
Y = np.array(Y)

# Reshape
X = X.reshape((X.shape[0], X.shape[1], 1))
X = X / len(chars)

# LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(seq_length,1)),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, Y, epochs=50)

# Prediction
test = "hello"

x_input = []

for ch in test:
    x_input.append(char_to_idx[ch])

x_input = np.array(x_input)
x_input = x_input.reshape((1, seq_length, 1))
x_input = x_input / len(chars)

prediction = model.predict(x_input)

predicted_char = idx_to_char[np.argmax(prediction)]

print("Input:", test)
print("Predicted Character:", predicted_char)
