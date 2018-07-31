# === dev env ===
# python 3.6

# === dependency ===
# keras
# h5py
# numpy

# === contact ===
# t105598018+wirl@ntut.org.tw

#
# load
#
import keras
model = keras.models.load_model('model.h5')


import numpy as np
#input : numpy array, batch*28*28*1, range: [0,1] 
canvas = np.zeros((1,28,28,1)) # first 1 is batch size

#predict (batch)
y_batch = model.predict(canvas) # batch predict

print(y_batch)
