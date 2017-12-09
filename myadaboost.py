import os
from PIL import Image
import sys
from numpy import *
from copy import *
#predict[49][286],classlabel[286]
#weight:sample initial weight
def find_classifier(mypredict,myclassLabel,weight):
	classifier_num,data_num=shape(mypredict)

	SelectedLabel=mat(zeros((data_num,1)))
	labelMatrix=mat(myclassLabel).T
	minError=10
	SelectedClassifier=0
	for i in range(0,classifier_num):
		predictVal=mat(mypredict[i]).T
		errArr=mat(ones((data_num,1)))
		errArr[predictVal==labelMatrix]=0
		weightedError=weight.T*errArr
		# print('classifier'+str(i)+': ')
		# print(errArr)
		# print(weightedError)
		if weightedError<minError and weightedError>0:
			# print('weighted error')
			# print(weightedError)
			# print(errArr.shape)
			# print(sum(errArr))
			minError=weightedError
			SelectedLabel=predictVal
			SelectedClassifier=i
	# print('selected')
	# print(SelectedClassifier)
	# # print('error: '+minError)
	return SelectedClassifier,minError,SelectedLabel
#here selected means the selected weak classifier

#predict[49][286] classLabel[286]
def adaBoostTrain(mypredict,myclassLabel):
	# classLabel=sign(myclassLabel-0.5)
	# predict=sign(mypredict-0.5)
	classLabel=sign(myclassLabel-0.5)
	predict=sign(mypredict)	

	classifier_num,data_num=shape(predict)
	alphaList=zeros(classifier_num)
	weight=mat(ones((data_num,1))/data_num)
	finalLabel=mat(zeros((data_num,1)))
	NowPrediction=mat(zeros((data_num,1)))

	not_all_correct=classifier_num
	print('classifier_num')
	print(classifier_num)
	for i in range(classifier_num):
		if sum(predict[i]!=classLabel)==0:
			print(i)
			alphaList[i]=4
			not_all_correct=not_all_correct-1
	print('not_all_correct')
	print(not_all_correct)
	for i in range(not_all_correct):
		SelectedClassifier,error,SelectedLabel=find_classifier(predict,classLabel,weight)
		# print('error:  '+str(error))
		if(error ==0):
			alpha=float(0.5*log((1-0.00001)/0.00001))
		else :
			alpha=float(0.5*log((1-error)/error))
		# print('error')
		# print(error)
		# print('alpha')
		# print(alpha)
		alphaList[SelectedClassifier]=alpha
		# print('shapetest: ')
		# print(classLabel.shape)
		# print(SelectedLabel.shape)
		expon=multiply(-1*alpha*mat(classLabel).T,SelectedLabel)
		# print(expon.shape)
		# print('expon: '+str(expon))
		weight=multiply(weight,exp(expon))
		weight=weight/weight.sum()
		finalLabel+=alpha*SelectedLabel

		aggErrors = multiply(sign(finalLabel) != mat(classLabel).T,ones((data_num,1)))		
		errorRate=aggErrors.sum()/data_num
		print('errRate: '+str(errorRate))
		if errorRate==0.0:
			break
	# print(alphaList)
	return alphaList,finalLabel