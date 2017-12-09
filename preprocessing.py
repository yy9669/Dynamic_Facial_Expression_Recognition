import os
from PIL import Image
import numpy as np
import sys
import math
from sklearn import preprocessing as pproc

##alignment and chopping in 1.py
def load_data():
	length=168
	width=136
	pixels=length*width
	allface=327
	patchsize=24
	patchpixels=patchsize*patchsize

	data=[]
	for i in range(0,80):
		data.append(np.empty((allface,6,patchpixels)))	
	label=[]
	#ff=open("/home/xuyang/6photolabel.txt")
	ff=open("label.txt")
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
		img=Image.open("./cropped/"+str(ii)+".bmp").convert('L')
		#img=Image.open("/media/fangxuyang/F/firefoxdownload/CK+database/equaugcropped/"+str(ii)+".bmp")
		#img=img.resize((46,56))
		#img=img.resize((width,length))
		index=i%6
		imgnum=i/6
		arr=np.asarray(img)
		arr=pproc.scale(arr)
		for j in range(0,80):
			leftcornerx=int(j/8)*16
			leftcornery=(j%8)*16
			for q in range(0,patchsize):
				for p in range(0,patchsize):
					data[j][imgnum][index][q*patchsize+p]=arr[leftcornerx+q][leftcornery+p]


	return data,label


#data,label=load_data()
#print(data)


