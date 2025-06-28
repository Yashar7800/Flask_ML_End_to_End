from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
import pycaret
from pycaret.regression import *

app = Flask(__name__)

model = load_model('gbr')

cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])

def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final],columns=cols)
    prediction = predict_model(model,data= data_unseen,round = 0)
    prediction = int(prediction['prediction_label'])
    return render_template('home.html',pred = 'Expected bill will be: {}'.format(prediction))

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data = data_unseen)
    output = prediction['prediction_label']
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

