from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained .h5 model
model = load_model('model.h5')  # Replace with your model's path

# Define the class labels
class_labels = ['Defective', 'Non-defective']

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Preprocess the image
    img = image.load_img(file, target_size=(224, 224))  # adjust size to your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize if your model requires it

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[int(predictions[0] > 0.5)]  # assumes binary classification

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
