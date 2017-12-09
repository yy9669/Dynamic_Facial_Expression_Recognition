import theano
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import merge
from keras.callbacks import EarlyStopping
from keras.regularizers import WeightRegularizer
from keras.regularizers import l1l2
import numpy as np
import random
from preprocessing2 import load_data
from keras.optimizers import RMSprop
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
newpredictresult=open("/home/xuyang/newpredictresult.txt", "w+")
data, label=load_data()
totalface=327
pixels=136*168

#shuffle the data
index=[]
for i in range(0,totalface):
	index.append(int(i))
random.shuffle(index)
sdata=np.empty((totalface,6,pixels))
slabel=[]
for i in range(0,totalface):
	sdata[i]=data[index[i]]
	slabel.append(label[index[i]])

testface=totalface/8+1;

#8 runs
totala=[0,0,0,0,0,0,0]
totalb=[0,0,0,0,0,0,0]
totalf1=[0,0,0,0,0,0,0]
a=0
b=0
f1=0
for q in range(0,8):
	if q!=7 and q!=0:
		testdata=sdata[q*testface:(q+1)*testface]
		testlabel=slabel[q*testface:(q+1)*testface]
		traindata=np.delete(sdata,np.s_[q*testface:(q+1)*testface],0)
		trainlabel=np.delete(slabel,np.s_[q*testface:(q+1)*testface],0)
	elif q==7:
		testdata=sdata[q*testface:]
		testlabel=slabel[q*testface:]
		traindata=sdata[:q*testface]
		trainlabel=slabel[:q*testface]
	else:
		testdata=sdata[:(q+1)*testface]
		testlabel=slabel[:(q+1)*testface]
		traindata=sdata[(q+1)*testface:]
		trainlabel=slabel[(q+1)*testface:]

	for p in range(0,7):

		model=Sequential()
		model.add(LSTM(128,input_shape=(6, pixels)))
		# model.add(Dense(128))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
		early_stopping = EarlyStopping(monitor='val_loss', patience=6)
		model.fit(np.array(traindata),np.array([y[p] for y in trainlabel]),nb_epoch=200,batch_size=32,shuffle=True,verbose=1,validation_split=0.1, callbacks=[early_stopping])
		yy=model.predict(testdata)

		truetest=0
		predicttrue=0
		predicttrueright=0
		predictwrong=0
		predictwrongright=0
		for i in range(0,len(testdata)):
			if testlabel[i][p]==1:
				truetest+=1
			if yy[i][0]>=0.5:
				predicttrue+=1
				if testlabel[i][p]==1:
					predicttrueright+=1
			if yy[i][0]<0.5:	
				predictwrong+=1
				if testlabel[i][p]==0:
					predictwrongright+=1	

		if predicttrue!=0:
			a=float(predicttrueright)/float(predicttrue)
		else:
			a=0
			newpredictresult.write("truetest:"+str(p)+" "+str(truetest)+"\n")

		if truetest!=0:
			b=float(predicttrueright)/float(truetest)
		else:
			b=0
			newpredictresult.write("predicttrue:"+str(p)+" "+str(predicttrue)+"\n")

		if a+b!=0:
			f1=2*a*b/(a+b)
		else:
			f1=0
		totala[p]+=a
		totalb[p]+=b
		totalf1[p]+=f1

for p in range(0,7):
	totala[p]=totala[p]/8
	totalb[p]=totalb[p]/8
	totalf1[p]=totalf1[p]/8
	thiskindface=0
	for i in range(0,totalface):
		if slabel[i][p]==1:
			thiskindface+=1
	newpredictresult.write("this kind of face num: "+str(thiskindface)+"\n")
	newpredictresult.write("Precision: "+str(totala[p])+"\n")
	newpredictresult.write("Recall: "+str(totalb[p])+"\n")
	newpredictresult.write("F1 score: "+str(totalf1[p])+"\n")
	newpredictresult.write("\n")


