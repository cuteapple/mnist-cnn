import numpy as np
import cv2
import keras

model = keras.models.load_model('model.h5')
canvas = np.zeros((28,28,1))
y = model.predict(canvas.reshape(1,28,28,1))[0]
print(y)