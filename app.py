from src import preprocess
import src.pdlload as pdlload
import os
import tensorflow as tf
from flask import Flask, request, render_template

my_path = os.path.abspath(os.path.dirname(__file__))


#print(data)
#data = pd.read_json(my_path + '/combined_dataset_20210930.json')#.loc[[0]].reset_index(drop=True)
#print()


model = tf.keras.models.load_model(my_path + '/model')

def pred(pred_data):
    return model.predict(pred_data)

# build the Flask app
app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def home():
    if request.method == "GET":
       return render_template("index.html",
                           value = 'david iyamah accountstory')
       
    if request.method == 'POST':
        company_and_name = request.form.get("name_employee_field")
        try:
            data = pdlload.call_enrich_api(company_and_name)
            data = preprocess.process_data(data)
            res = pred(data)
            res = round(res[0][0] * 100, 1)
        except:
            res = -1
        
        
        return render_template("index.html", value = res, company_and_name = company_and_name)
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)