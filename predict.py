import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model('indian_food_model.h5')

# Test image path
img_path = 'test4.jpg'  # Replace with your actual image
img = image.load_img(img_path, target_size=(224, 224))

# Preprocess
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)

# Load class labels
class_names = sorted(os.listdir('dataset'))
predicted_class = class_names[np.argmax(predictions)]

print(f"🍽️ Predicted Food: {predicted_class}")
