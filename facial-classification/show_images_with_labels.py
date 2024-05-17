import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Define model filename
model_filename = "models/faces_model_all_shuffled.keras"
img_height = img_width = 222
sample_size = 6
# Read CSV file
df = pd.read_csv('data.csv', sep=',', header=None, names=['Counting', 'Path', 'Label'])

# Extract paths and labels
paths = df['Path'].values[1:]
labels = df['Label'].values[1:]

# Randomly select 5 indices
random_indices = np.random.choice(len(paths), size=sample_size, replace=False)

# Load and preprocess the selected images
selected_images = []
selected_labels = []

for index in random_indices:
    path = "dataset\\" + paths[index]
    if os.path.exists(path):
        image = Image.open(path)
        image = image.resize((img_width, img_height))  # Resize image
        image = np.array(image).astype('float32') / 255.0  # Normalize pixel values
        selected_images.append(image)
        selected_labels.append(labels[index])
    else:
        print(f"File not found: {path}")

# Convert selected images and labels to numpy arrays
selected_images = np.array(selected_images)
selected_labels = np.array(selected_labels)

# Check if the model file exists
if os.path.exists(model_filename):
    # Load the existing model
    model = tf.keras.models.load_model(model_filename)
    print("Loaded existing model")
else:
    print("Model file not found.")

# Define the class mapping for specific classes
class_mapping = {
    'Ahegao': 0,
    'Angry': 1,
    'Happy': 2,
    'Neutral': 3,
    'Sad': 4,
    'Surprise': 5
}

# Make predictions
predictions = model.predict(selected_images)
predicted_labels = np.argmax(predictions, axis=1)

# Decode labels
predicted_labels = [list(class_mapping.keys())[list(class_mapping.values()).index(label)] for label in predicted_labels]

# Display the images along with their predicted and actual labels
plt.figure(figsize=(15, 8))
for i in range(sample_size):
    plt.subplot(1, sample_size, i+1)
    plt.imshow(selected_images[i])
    plt.title(f'Predicted: {predicted_labels[i]}\nActual: {selected_labels[i]}')
    plt.axis('off')
plt.show()
print()
