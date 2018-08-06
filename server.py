from flask import Flask, jsonify, request, send_file
import numpy as np
#import keras

app = Flask(__name__)
#model = keras.models.load_model('model.h5')

@app.route('/')
def index():
	return send_file('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	json = request.get_json(force=True)
	return jsonify(pred = [float(x) for x in np.random.rand(10)])

app.run()