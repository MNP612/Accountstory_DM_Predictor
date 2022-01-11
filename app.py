import pandas as pd
import numpy as np
#import preprocess
import os
import tensorflow as tf
from flask import Flask, request, render_template

my_path = os.path.abspath(os.path.dirname(__file__))

#data = pd.read_json(my_path + '/combined_dataset_20210930.json')#.loc[[0]].reset_index(drop=True)
#data = preprocess.process_data(data)

model = tf.keras.models.load_model(my_path + '/model')

def pred(pred_data):
    return model.predict(pred_data)

# build the Flask app
app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def home():
    if request.method == "GET":
       return render_template("index.html",
                           value = 1)
       
    if request.method == 'POST':
        employee_index = request.form.get("name_employee")
        #pred_data = np.array(data)[int(employee_index)][:314].astype(float).reshape(1,-1)
        pred_data = np.zeros((1,314))
        res = pred(pred_data)
        res = round(res[0][0], 2)
        return render_template("index.html",
                           value = res)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)