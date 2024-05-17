import pandas as pd
import tempfile
import os
import pandas as pd
from PIL import Image
import numpy as np
from keras.src.utils import to_categorical
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split


img_height = img_width = 222


def get_data():
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
    # # take 10 random sampeles from paths and labels
    # paths = paths[:300]
    # labels = labels[:300]
    # Load and preprocess images
    images = []
    for i, path in enumerate(paths):
        if i % 500 == 0:
            print(f"open image {i}")
        path = "dataset\\" + path
        if os.path.exists(path):  # Check if the file exists
            image = Image.open(path)
            # Convert image to black and white (grayscale)
            image = image.convert('L')
            image = image.resize((img_width, img_height))  # Resize image
            # Normalize pixel values to the range [0, 1]
            image = np.array(image)/ 255.0
            images.append(image)
        else:
            print(f"File not found: {path}")

    # Convert the list of images to a NumPy array
    images = np.array(images, dtype=np.float32)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)  # One-hot encode labels

    # Split dataset 15%, 15%, 75%
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=42)

    print("split the data")
    return X_train, X_test, y_train, y_test

def data_rgb():
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
    # # take 10 random sampeles from paths and labels
    # paths = paths[:1000]
    # labels = labels[:1000]
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
    X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

    print("split the data")
    return X_train, X_test, y_train, y_test