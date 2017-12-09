import os
from PIL import Image
import numpy as np
import sys
import math
from sklearn import preprocessing

##alignment and chopping in 1.py

def load_data():
	length=168
	width=136
	pixels=length*width
	allface=327

	data=[]
	data=np.empty((allface,6,pixels))
	label=[]
	#ff=open("/home/xuyang/6photolabel.txt")
	ff=open("/home/xuyang/label.txt")
	#ff=open("/media/fangxuyang/F/firefoxdownload/CK+database/dataaug/6photolabel.txt")
	dd=ff.read()
	totallabel=dd.split('\n')

	for i in range(0,allface):
		facelabel=[0,0,0,0,0,0,0]
		facelabel[int(totallabel[i+1])-1]=1
		label.append(facelabel)

	for i in range(0,6*allface):
		ii=i+1
		#img=Image.open("/home/xuyang/equaugcropped/"+str(ii)+".bmp")
		img=Image.open("/home/xuyang/cropped/"+str(ii)+".bmp").convert('L')
		#img=Image.open("/media/fangxuyang/F/firefoxdownload/CK+database/equaugcropped/"+str(ii)+".bmp")
		#img=img.resize((46,56))
		#img=img.resize((width,length))
		index=i%6
		imgnum=i/6
		arr=np.asarray(img)
		arr=preprocessing.scale(arr)
		for q in range(0,length):
			for p in range(0,width):
				data[imgnum][index][q*width+p]=arr[q][p]
	return data,label

#data,label=load_data()
#print(data)

