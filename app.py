
# -*- coding: utf-8 -*-


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_mobnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    result = model.predict(x)
    result=result.tolist()
    names={0:'Ford',
 1:'Honda',
 2:'Hyundai',
 3:'Jeep',
 4:'Kia',
 5:'MG',
 6:'Mahindra',
 7:'Maruti',
 8:'Nissan',
 9:'Renault',
 10:'Skoda',
 11:'Tata',
 12:'Toyota',
 13:'Volkswagen'}

    
    
    return names[result[0].index(max(result[0]))]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'Uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return render_template('index.html', prediction_text='It belongs to {}'.format(result))
    return None


if __name__ == '__main__':
    app.run(debug=True)
