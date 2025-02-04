# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:21:53 2025

@author: pradi
"""


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Path 

image_dir = "C:/Users/pradi/Downloads/archive/1920x1080/1920x1080/train"
mask_dir = "C:/Users/pradi/Downloads/archive/1980x1080/labels/labels/areas/train_labels_1920x1080"
test_image_dir = "C:/Users/pradi/Downloads/archive/1920x1080/1920x1080/test"
test_mask_dir = "path_to_test_masks"

# Parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Resizing images
NUM_CLASSES = 4  # Background + other classes from the mask (values: 0, 1, 3)

# Function to preprocess images
def preprocess_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(img) / 255.0  # Normalize to [0, 1]

# Function to preprocess masks
def preprocess_mask(mask_path):
    mask = Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.NEAREST)
    mask_array = np.array(mask)
    return tf.keras.utils.to_categorical(mask_array, num_classes=NUM_CLASSES)

# Load dataset
def load_dataset(image_dir, mask_dir):
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)  # Assuming matching names
        if os.path.exists(mask_path):
            images.append(preprocess_image(img_path))
            masks.append(preprocess_mask(mask_path))

    return np.array(images), np.array(masks)

# Load test dataset
def load_test_data():
    test_images, test_masks = load_dataset(test_image_dir, test_mask_dir)
    return test_images, test_masks

# Define the MobileNetV3-based U-Net model
def build_model():
    base_model = MobileNetV3Large(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Encoder (Feature extraction)
    encoder_output = base_model.output

    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer
    output = layers.Conv2D(NUM_CLASSES, (1, 1), activation="softmax")(x)

    return models.Model(inputs=base_model.input, outputs=output)

# Compile the model
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Evaluation and visualization
def evaluate_model(model, test_images, test_masks):
    predictions = model.predict(test_images)
    predictions = np.argmax(predictions, axis=-1)  # Convert from one-hot to class indices
    ground_truth = np.argmax(test_masks, axis=-1)

    # Example metric: Pixel-wise accuracy
    pixel_accuracy = np.mean(predictions == ground_truth)

    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")

    return predictions

def display_predictions(images, masks, predictions, num_samples=3):
    for i in range(num_samples):
        original_image = images[i]
        ground_truth_mask = np.argmax(masks[i], axis=-1)
        predicted_mask = predictions[i]

        plt.figure(figsize=(15, 5))

        # Image originale
        plt.subplot(1, 3, 1)
        plt.title("Image Originale")
        plt.imshow(original_image)
        plt.axis("off")

        # Masque attendu
        plt.subplot(1, 3, 2)
        plt.title("Masque Attendu")
        plt.imshow(ground_truth_mask, cmap="jet", alpha=0.7)
        plt.axis("off")

        # Masque prédit
        plt.subplot(1, 3, 3)
        plt.title("Masque Prédit")
        plt.imshow(original_image)
        plt.imshow(predicted_mask, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.show()

evaluate_model(model, load_dataset(image_dir, mask_dir)[0],load_dataset(image_dir, mask_dir)[1])
"""
# Example usage
# Charger les données de test
test_images, test_masks = load_test_data()

# Charger les poids entraînés
model.load_weights("path_to_saved_weights.h5")

# Générer des prédictions
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=-1)  # Convertir en indices de classe

# Afficher les résultats
display_predictions(test_images, test_masks, predictions, num_samples=5)

# Parameters
IMG_HEIGHT, IMG_WIDTH = 256, 256  # Resizing images
NUM_CLASSES = 4  # Background + other classes from the mask (values: 0, 1, 3)

# Function to preprocess images
def preprocess_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    return np.array(img) / 255.0  # Normalize to [0, 1]

# Function to preprocess masks
def preprocess_mask(mask_path):
    mask = Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.NEAREST)
    mask_array = np.array(mask)
    return tf.keras.utils.to_categorical(mask_array, num_classes=NUM_CLASSES)

# Load dataset
def load_dataset(image_dir, mask_dir):
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)  # Assuming matching names
        if os.path.exists(mask_path):
            images.append(preprocess_image(img_path))
            masks.append(preprocess_mask(mask_path))

    return np.array(images), np.array(masks)

# Load the dataset (Replace paths with real ones)
# images, masks = load_dataset(image_dir, mask_dir)

# Define the MobileNetV3-based U-Net model
def build_model():
    base_model = MobileNetV3Large(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Encoder (Feature extraction)
    encoder_output = base_model.output

    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(encoder_output)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer
    output = layers.Conv2D(NUM_CLASSES, (1, 1), activation="softmax")(x)

    return models.Model(inputs=base_model.input, outputs=output)

# Compile the model
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training (example code, replace with real dataset)
# model.fit(images, masks, batch_size=16, epochs=20, validation_split=0.2)

import matplotlib.pyplot as plt
def load_test_data():
    test_images, test_masks = load_dataset(test_image_dir, test_mask_dir)
    return test_images, test_masks

# Fonction pour afficher les images, masques et prédictions
def display_predictions(images, masks, predictions, num_samples=3):
    for i in range(num_samples):
        original_image = images[i]
        ground_truth_mask = np.argmax(masks[i], axis=-1)
        predicted_mask = predictions[i]

        plt.figure(figsize=(15, 5))

        # Image originale
        plt.subplot(1, 3, 1)
        plt.title("Image Originale")
        plt.imshow(original_image)
        plt.axis("off")

        # Masque attendu
        plt.subplot(1, 3, 2)
        plt.title("Masque Attendu")
        plt.imshow(ground_truth_mask, cmap="jet", alpha=0.7)
        plt.axis("off")

        # Masque prédit
        plt.subplot(1, 3, 3)
        plt.title("Masque Prédit")
        plt.imshow(original_image)
        plt.imshow(predicted_mask, cmap="jet", alpha=0.5)
        plt.axis("off")

        plt.show()

# Exemple d'utilisation
# Charger les données de test
test_images, test_masks = load_test_data()

# Charger les poids entraînés
model.load_weights("path_to_saved_weights.h5")

# Générer des prédictions
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=-1)  # Convertir en indices de classe

# Afficher les résultats
display_predictions(test_images, test_masks, predictions, num_samples=5)"""