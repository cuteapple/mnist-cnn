# === dev env ===
# python 3.6

# === dependency ===
# h5py
# keras
# ==================
# assume tensorflow backend (data)

# === contact ===
# t105598018+wirl@ntut.org.tw
import keras

#
# create model
#
from keras.models import Sequential
from keras.layers import Conv2D, Dropout,Dense,Flatten

layers = [Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)),
	Conv2D(64,kernel_size=3,activation='relu'),
	Conv2D(128,kernel_size=3,strides=2,activation='relu'),
	Dropout(0.25),

	Conv2D(192,kernel_size=3,activation='relu'),
	Conv2D(192,kernel_size=3,strides=2,activation='relu'),
	Dropout(0.25),

	Conv2D(256,(3,3),activation='relu'),
	Dropout(0.25),

	Flatten(),
	Dense(256),
	Dropout(0.5),

	Dense(10, activation='softmax')]

model = Sequential(layers,name='mnist')

model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy','mae'])

#
# load datas
#
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
(x,y),(tx,ty) = mnist.load_data()

x = x.astype(float).reshape(-1,28,28,1) / 255
tx = tx.astype(float).reshape(-1,28,28,1) / 255

y = keras.utils.to_categorical(y,10)
ty = keras.utils.to_categorical(ty,10)

g = ImageDataGenerator(width_shift_range=5,height_shift_range=5)

#
# train
#
#model.fit(x,y,batch_size=32,epochs=100,validation_data=(tx,ty))
model.fit_generator(g.flow(x,y,batch_size=32),steps_per_epoch=len(x) // 32,epochs=100,validation_data=(tx,ty))
model.save('model.h5')