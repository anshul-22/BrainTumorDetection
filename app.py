from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Determine the path of the current script
current_path = os.path.dirname(os.path.realpath(__file__))

# Set the static and template folder paths
static_folder = os.path.join(current_path, 'static')
template_folder = os.path.join(current_path, 'templates')

# Initialize Flask app with custom static and template folder paths
app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)

# Load the model
model_path = os.path.join(current_path, 'brain_tumor_model.h5')
model = load_model(model_path)

# Preprocess image function
def preprocess_image(image):
    # Resize image to match model input shape
    resized_image = cv2.resize(image, (240, 240))
    # Ensure the image has a single channel (grayscale)
    if len(resized_image.shape) == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Add channel dimension
    processed_image = np.expand_dims(resized_image, axis=-1)
    # Normalize image
    processed_image = processed_image / 255.0
    return processed_image


# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            # Read and preprocess the uploaded image
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            processed_image = preprocess_image(image)
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

            # Make prediction using the model
            prediction = model.predict(processed_image)
            result = "!! Brain tumor detected !!" if prediction >= 0.5 else "No brain tumor detected!"

            return render_template('result.html', result=result)
        else:
            return "Error: No image uploaded!"

if __name__ == '__main__':
    # Run the app
    app.run(debug=True)
