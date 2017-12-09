import theano
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation,Flatten, Reshape
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import merge
import numpy as np
from preprocessing import load_data
from myadaboost import find_classifier
from myadaboost import adaBoostTrain
from keras import backend as K


totalface=327
pixels=136*168
patchsize=24
patchnum=80
totaltrainface=280
patchpixels=patchsize*patchsize

rawdata, rawlabel=load_data()
traindata=np.array([y[0:totaltrainface] for y in rawdata])#[80][280]

AllSavedModel=[]
for i in range(0,patchnum):
	savedweight="./boostedresult/my_model"+str(i)+".h5"
	AllSavedModel.append(savedweight)

model=load_model(AllSavedModel[20])
print(traindata[20])
print("yes")



OneModelPredict=model.predict(traindata[20])
print(OneModelPredict)