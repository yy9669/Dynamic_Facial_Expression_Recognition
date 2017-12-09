import os
import cv2
import numpy as np
from keras.utils import np_utils, generic_utils
from sklearn import preprocessing

def load_data():
	print("start loading data...")
	length = 168 # 168/24 = 7
	patch_size = 24
	patch_num_per_row = int(length / patch_size)
	patch_num_per_col = patch_num_per_row
	width = 136
	allface = 327
	series_num = 6
	express_num = 7
	pixels = length * width
	data = np.empty((allface,series_num, patch_num_per_row, patch_num_per_col, patch_size, patch_size, 1), dtype = "float32")
	#data = np.empty((allface,6,length,length,1), dtype = "float32")
	label = []
	myfile = open("./label/label.txt")
	content = myfile.read()
	total_label = content.split('\n')

	for i in range(0, allface):
		label.append(int(total_label[i + 1]) - 1)

	for i in range(0, series_num * allface):
		imgnum = int(i/6)
		imgindex = i%6
		img = cv2.imread("./data/" + str(i+1) + ".bmp", 0) # gray img
		img = cv2.resize(img, (length, length), interpolation = cv2.INTER_CUBIC)
		#img = cv2.equalizeHist(img)
		img = preprocessing.scale(img)
		for row in range(0, patch_num_per_row):
			for col in range(0, patch_num_per_col):
				data[imgnum,imgindex,row,col,:,:,:] = img[row*patch_size:(row+1)*patch_size,col*patch_size:(col+1)*patch_size,np.newaxis]
		#data[imgnum,imgindex,:,:,:] = img[:, :, np.newaxis]
	label = np_utils.to_categorical(label, express_num)
	print("finished load data...")
	return data, np.array(label)

if __name__ == "__main__":
	data, label = load_data()
	print(data.shape, label.shape)
	print(label)
