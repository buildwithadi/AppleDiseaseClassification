import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt




# Initialize the Flask application
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_names = ['Scab', 'Rotten', 'Cedar', 'Normal']

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array) 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def make_prediction(image_path):

    with open('models/vgg16Apple.pkl', 'rb') as f:
        model = pickle.load(f)
    img_tensor = load_and_preprocess_image(image_path)
    prediction = model.predict(img_tensor)
    
    predicted_class = np.argmax(prediction)

    return predicted_class


# --- FLASK ROUTES ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return redirect(request.url) # Redirect if no file part

    file = request.files['image_file']

    if file.filename == '':
        return redirect(request.url) # Redirect if no file is selected

    if file:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get the prediction from your model
        prediction_result = make_prediction(filepath)

        # Pass the result back to the HTML template
        return render_template('index.html', prediction_text='The Apple is ' + class_names[prediction_result])

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)