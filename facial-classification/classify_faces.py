import os
import keras
import pandas as pd
from PIL import Image
import numpy as np
from keras import layers
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

# Define model filename
model_filename = f"models/faces_model_all_shuffled.keras"
img_height = img_width = 222

# Read CSV file
df = pd.read_csv('data.csv', sep=',', header=None, names=['Counting', 'Path', 'Label'])

# Extract paths and labels
paths = df['Path'].values[1:]
labels = df['Label'].values[1:]

# Combine paths and labels into a list of tuples
data = list(zip(paths, labels))

# Shuffle the list
np.random.shuffle(data)

# Split back into paths and labels
paths, labels = zip(*data)

# Load and preprocess images
images = []
for i, path in enumerate(paths):
    if i % 500 == 0:
        print(f"open image {i}")
    path = "dataset\\" + path
    if os.path.exists(path):  # Check if the file exists
        image = Image.open(path)
        image = image.resize((img_width, img_height))  # Resize image
        image = np.array(image)
        images.append(image)
    else:
        print(f"File not found: {path}")

# Convert the list of images to a NumPy array
images = np.array(images)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_encoded = to_categorical(labels_encoded)  # One-hot encode labels

# Split dataset 15%, 15%, 75%
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=42)

print("split the data")

# Check if the model file exists
if os.path.exists(model_filename):
    # Load the existing model
    model = tf.keras.models.load_model(model_filename)
    print("Loaded existing model")
else:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )
    # Define the neural network architecture
    model = keras.models.Sequential([

        data_augmentation,

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),

        layers.Dense(64, activation='relu'),

        layers.Dense(len(np.unique(labels)), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

epochs = 10
# Train the model
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

# Save the trained model
model.save(model_filename)
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# Randomly select 5 indices from the test dataset
random_indices = np.random.choice(len(X_test), size=5, replace=False)

# Retrieve the corresponding images and labels
selected_images = X_test[random_indices]
selected_labels = y_test[random_indices]

# Make predictions
predictions = model.predict(selected_images)
predicted_labels = np.argmax(predictions, axis=1)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predicted_labels)
actual_labels = label_encoder.inverse_transform(np.argmax(selected_labels, axis=1))

# Display the images along with their predicted and actual labels
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(selected_images[i])
    plt.title(f'Predicted: {predicted_labels[i]}\nActual: {actual_labels[i]}')
    plt.axis('off')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

