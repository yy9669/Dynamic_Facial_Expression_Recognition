from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import merge
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
import random
from keras.optimizers import RMSprop
from load_data_patch import load_data


#some arguments
total_size = 327
train_size = 286
#327 = 286 + 41
test_size = total_size - train_size
patch_size = 24
patch_pixels = patch_size * patch_size
length = 168
width = length
total_pixels = length * width
series_num = 6
expression_num = 7
patch_per_row = int(length / patch_size)
patch_per_col = int(width / patch_size)


def load_random():
	tmpdata, tmplabel = load_data()
	data = tmpdata.copy()
	label = tmplabel.copy()
	index = []
	with open("./random_index.txt") as f:
		index = f.read().split('\n')
	for i in range(0, total_size):
		data[i] = tmpdata[int(index[i])]
		label[i] = tmplabel[(int(index[i]))]

	return data,label

def binary_train():
	#folder from 0 to 7 since we use cross validation way here
	for folder in range(0,8):
		work_dir = "./model/binary/" + str(folder)
		data, label = load_random()
		train_data = data[:train_size].copy()
		train_label = label[:train_size].copy()
		test_data = data[train_size:].copy()
		test_label = label[:train_size].copy()
		#data prepare for next folder----
		tmp = data[(folder)*test_size:(folder+1)*test_size].copy()
		data[(folder)*test_size:(folder+1)*test_size] = data[train_size:]
		data[train_size:] = tmp[:]
		#--------------------------------

		for row in range(0, patch_per_row):
			for col in range(0, patch_per_col):
				for i in range(0, expression_num):
					classifier_id = row*patch_per_row*patch_per_col+col*patch_per_col+i
					print("Constructing "+str(classifier_id)+"th classifier.")
					pixel_train_label = []
					for l in train_label:
						if l[i] == 1:
							pixel_train_label.append(1)
						else:
							pixel_train_label.append(0)

					pixel_train_data = train_data[:,:,row,col,:,:,:]
					pixel_train_data = np.reshape(pixel_train_data, (train_size,series_num,patch_pixels))

					#-------model definition-------
					inputs = Input(shape = (series_num, patch_pixels))
					lstm1 = LSTM(128)
					x = lstm1(inputs)
					fc1 = Dense(1,activation = "sigmoid")
					predictions = fc1(x)
					model = Model(input = inputs, output = predictions)
					model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ['accuracy'])
					#early_stopping = EarlyStopping(monitor='val_loss', patience = 6)
					#model.fit(pixel_traindata, pixel_trainlabel, nb_epoch = 200, validation_split = 0.125, shuffle = False, callbacks = [early_stopping])
					model.fit(pixel_train_data, pixel_train_label, nb_epoch = 200, validation_split = 0.1)
					model.save(work_dir+"/"+"binary_model"+str(classifier_id)+".h5")
	return

if __name__ == "__main__":
	binary_train()




