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
import os
import tensorflow as tf
from keras.optimizers import RMSprop
from load_data_patch import load_data
from binary_train import load_random
import keras.backend.tensorflow_backend as KTF

def set_gpu(gpu_fraction = 0.4):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_fraction)
	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
	KTF.set_session(sess)

#some arguments
total_size = 327
train_size = 286
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

def load_binary_model(workdir):
	print("loading model in " + workdir)
	all_model = []
	for row in range(0, patch_per_row):
			for col in range(0, patch_per_col):
				for i in range(0, expression_num):
					classifier_id = row*patch_per_row*patch_per_col+col*patch_per_col+i
					model = load_model(workdir+"/"+"binary_model"+str(classifier_id)+".h5")
					#model.trainable = False
					all_model.append(model)
	print("loading models in " + workdir + "is finished")
	return all_model

def Boost_FC():
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
		all_train_data = []
		all_train_label = []
		for row in range(0, patch_per_row):
			for col in range(0, patch_per_col):
				for i in range(0, expression_num):
					pixel_train_data = train_data[:,:,row,col,:,:,:]
					pixel_train_data = np.reshape(pixel_train_data, (train_size, series_num, patch_pixels))
					all_train_data.append(pixel_train_data)
		all_model = load_binary_model(work_dir)
		all_train_label = train_label
		print("all models are loaded")


		all_input = []
		all_tmp = []
		for i in range(0, patch_per_row*patch_per_col*expression_num):
			all_input.append(Input(shape = (series_num, patch_pixels)))
			tmp = (all_model[i])(all_input[i])
			all_tmp.append(tmp)
		merged_tmp = merge(all_tmp, mode = 'concat', concat_axis = -1)
		predictions0 = Dense(128, activation = "relu")(merged_tmp)
		predictions = Dense(expression_num, activation = "softmax")(predictions0)
		model = Model(input = all_input, output = predictions)
		model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics =["accuracy"])
		early_stopping = EarlyStopping(monitor = "val_loss", patience = 10)

		print("start training")
		model.fit(all_train_data, all_train_label, nb_epoch = 200, batch_size = 32, validation_split = 0.1, callbacks = [early_stopping])
		model.save(work_dir+"/"+"boost_fc_model"+".h5")
		print("training is finished and model has been saved.")


def Boost_indif():
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
		all_train_data = []
		all_train_label = []
		for row in range(0, patch_per_row):
			for col in range(0, patch_per_col):
				for i in range(0, expression_num):
					pixel_train_data = train_data[:,:,row,col,:,:,:]
					pixel_train_data = np.reshape(pixel_train_data, (train_size, series_num, patch_pixels))
					all_train_data.append(pixel_train_data)
		all_model = load_binary_model(workdir)
		all_train_label = train_label



		all_input = []
		all_tmp = []

		for i in range(0,expression_num):
			all_tmp.append([])

		for row in range(0, patch_per_row):
			for col in range(0, patch_per_col):
				for i in range(0, expression_num):
					classifier_id = row*patch_per_row*patch_per_col+col*patch_per_col+i
					all_input.append(Input(shape = (series_num, patch_pixels)))
					tmp = (all_model[classifier_id])(all_input[classifier_id])
					all_tmp[i].append(tmp)
		merged_tmp = []
		predictions0 = []
		predictions1 = []
		for i in range(0, expression_num):
			merged_tmp.append(merged(all_tmp[i], mode = 'concat', concat_axis = -1))
			predictions0.append(Dense(128, activation = "relu")(merged_tmp[i]))
			predictions1.append(Dense(1, activation = "sigmoid")(predictions0[i]))

		predictions = merged(predictions1, mode = 'concat', concat_axis = -1)
		model = Model(input = all_input, output = predictions)
		model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics =["accuracy"])
		early_stopping = EarlyStopping(monitor = "val_loss", patience = 10)
		model.fit(all_train_data, all_train_label, nb_epoch = 200, batch_size = 32, validation_split = 0.1, callbacks = [early_stopping])
		model.save(work_dir+"/"+"boost_fc_model"+".h5")

if __name__ == "__main__":
	print(1)
	set_gpu()
	print(2)
	Boost_FC()
	print(3)
	Boost_indif()