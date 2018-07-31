import keras

# load
model = keras.models.load_model('model.h5')


import numpy as np
#input : numpy array, batch*28*28*1, range: [0,1] 
canvas = np.zeros((2,28,28,1)) # 2 is batch size (2 images)

#predict (batch)
y_batch = model.predict(canvas) # batch predict

#can see 2 result
print(y_batch)