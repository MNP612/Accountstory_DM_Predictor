#import pandas as pd
#import numpy as np
#import preprocess
#import os
#import tensorflow as tf
#from tensorflow import keras

from flask import Flask, request, render_template

#data = pd.read_json('combined_dataset_20210930.json').loc[[100]].reset_index(drop=True)

#data = preprocess.process_data(data)
#data = np.zeros((1,314))

# class UnetInferrer:
#     def __init__(self):
#         self.data = data
#         #print('YOOOOOO', self.data)
#         self.saved_path = ''
#         self.model = tf.keras.models.load_model('model.h5')

#         self.prediction = self.model.predict(self.data)

#     def pred(self):
#         return self.prediction

# res = UnetInferrer().pred()

# build the Flask app
app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def home():
    # if request.method == "GET":
    #    return render_template("index.html",
    #                        value = '')
       
    # if request.method == 'POST':
    #     first_name = request.form.get("name_employee")
    #     return render_template("index.html",
    #                        value = first_name)
    return render_template("index.html")

#port = int(os.environ.get('PORT', 33507))
    
if __name__ == "__main__":
    app.run(debug=True)