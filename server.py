import keras
model = keras.models.load_model('model.h5')
model._make_predict_function()

from flask import Flask, jsonify, request, send_file
app = Flask(__name__)

import numpy as np
import cv2

@app.route('/')
def index():
	return send_file('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	json = request.get_json(force=True)
	data = np.array(json,dtype=float).reshape(1,28,28,1)

	cv2.imwrite('last-predict.png',data[0] * 255)
	return jsonify(pred = [float(x) for x in model.predict(data)[0]])

app.run()