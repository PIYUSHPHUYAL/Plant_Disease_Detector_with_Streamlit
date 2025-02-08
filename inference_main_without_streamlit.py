import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('plant_disease_cnn.h5')  # Replace with your model path

class_names = [
    "Apple_Black_Rot",
    "Apple_Healthy",
    "Apple_Scab",
    "Bell_Pepper_Healthy",
    "Bell_Pepper_Bacterial_Spot",
    "Cedar_Apple_Rust",
    "Cherry_Healthy",
    "Cherry_Powdery_Mildew",
    "Grape_Black_Rot",
    "Grape_Esca_(Black_Measles)",
    "Grape_Healthy",
    "Grape_Leaf_Blight",
    "Maize_Cercospora_Leaf_Spot",
    "Maize_Common_Rust",
    "Maize_Healthy",
    "Maize_Northern_Leaf_Blight",
    "Peach_Bacterial_Spot",
    "Peach_Healthy",
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight",
    "Strawberry_Healthy",
    "Strawberry_Leaf_Scorch",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Septoria_Leaf_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus"
]

def preprocess_image(img_path):
    # Load and resize the image to 256x256 (same as training)
    img = image.load_img(img_path, target_size=(256, 256))
    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    # Add batch dimension (shape = [1, 256, 256, 3])
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Example usage
image_path = "mai.JPG"  # Replace with your image
processed_image = preprocess_image(image_path)

# Get predictions
prediction = model.predict(processed_image)
predicted_class_index = np.argmax(prediction)
confidence = np.max(prediction) * 100
predicted_class = class_names[predicted_class_index]

# Print results
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
