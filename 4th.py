import tensorflow as tf

# Load dataset from folder
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(224,224),
    batch_size=32
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/val",
    image_size=(224,224),
    batch_size=32
)

# Load pretrained ResNet50 model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Create model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_data.class_names), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("resnet50_model.keras")
