import numpy as np
import cv2
import keras

sx,sy = 10,10

model = keras.models.load_model('model.h5')
canvas = np.zeros((28,28,1))
window = 'canvas'

cv2.namedWindow(window)

change = False
mousehold = False
def onmouse(event,x,y,flags,param):
	global mousehold,change

	if event == cv2.EVENT_LBUTTONDOWN:
		mousehold = True
		x,y = x // sx,y // sy

	elif event == cv2.EVENT_MOUSEMOVE:
		if not mousehold:
			return
		x,y = x // sx,y // sy

	elif event == cv2.EVENT_LBUTTONUP:
		mousehold = False
		return

	#print(y,x)
	if 0 < y < 27 and 0 < x < 27:
		change = change or canvas[y,x] != 1
		canvas[y,x] = 1

cv2.setMouseCallback(window,onmouse)

while True:
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		canvas = np.zeros((28,28,1))
	
	cv2.imshow(window,cv2.resize(canvas,(28 * sx,28 * sy),interpolation=cv2.INTER_NEAREST))
	if change:
		print(model.predict(canvas.reshape(1,28,28,1))[0])
		change = False