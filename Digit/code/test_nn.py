#!/usr/bin/env python
import matplotlib
import pybrain

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import TanhLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from sklearn import preprocessing
import csv_io_np
import csv_io
import math
from math import log

import numpy as np

import scipy

import gc

import datetime
import random

	
#http://pybrain.org/docs/tutorial/fnn.html	
	
def run_stack(SEED):

	# means = [(-1,0),(2,4),(3,1)]
	# cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
	# alldata = ClassificationDataSet(2, 1, nb_classes=3)

	# for n in xrange(400):
		# for klass in range(3):
			# input = multivariate_normal(means[klass],cov[klass])
			# print input, klass
			# alldata.addSample(input, [klass])

	# tstdata, trndata = alldata.splitWithProportion( 0.25 )

	# trndata._convertToOneOfMany( )
	# tstdata._convertToOneOfMany( )
	# print "Number of training patterns: ", len(trndata)
	# print "Input and output dimensions: ", trndata.indim, trndata.outdim
	# print "First sample (input, target, class):"
	# print trndata['input'][0], trndata['target'][0], trndata['class'][0]
	# fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
	# trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
	# for i in range(20):
		# trainer.trainEpochs( 1 )

		# trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
		# tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

		# print "epoch: %4d" % trainer.totalepochs, \
			  # "  train error: %5.2f%%" % trnresult, \
			  # "  test error: %5.2f%%" % tstresult	
			
	# return	
		
		

	trainBase = csv_io_np.read_data("PreProcessData/train.csv", skipFirstLine = True, split = ",")
	test = csv_io_np.read_data("PreProcessData/test.csv", skipFirstLine = True, split = ",")

	print "Data Read Complete"


	
	print "input Dim:" , len(trainBase[0]) - 1
	alldata = ClassificationDataSet(len(trainBase[0]) - 1, 1, nb_classes=10)
	
	#alldata.setField('input', [x[1:] for x in trainBase])
	#alldata.setField('target', [x[0] for x in trainBase])
	
	#scaler = preprocessing.Scaler()
	#scaler.fit([x[1:] for x in trainBase]) # .astype('double')
	#trainScaled = scaler.transform(X)
	trainScaled = trainBase
	
	
	count = 0
	for index, row in enumerate(trainScaled):
		#print count,  row[0]
		alldata.addSample(np.divide(row[1:], 255), row[0])
		#alldata.addSample(row[0:], trainBase[index,0])
		count += 1
		if ( count == 16000 ) :  ####### try to get more data in.....
			break
		
	gc.collect()	
		


	tstdata, trndata = alldata.splitWithProportion( 0.2 )

	gc.collect()
	
	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )
	
	gc.collect()
	
	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"
	#print trndata['input'][0], trndata['target'][0], trndata['class'][0]
	#print trndata['input'][1], trndata['target'][1], trndata['class'][1]	
	#print trndata['input'][2], trndata['target'][2], trndata['class'][2]
	#print trndata['input'][3], trndata['target'][3], trndata['class'][3]

	
	fnn = buildNetwork( trndata.indim, 800, trndata.outdim, hiddenclass=TanhLayer, outclass=SoftmaxLayer , bias=True) # num inputs, num hidden layer, num outputs.
	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.001)

	
	#ticks = arange(-3.,6.,0.2)
	#X, Y = meshgrid(ticks, ticks)
	## need column vectors in dataset, not arrays
	#griddata = ClassificationDataSet(2,1, nb_classes=3)
	#for i in xrange(X.size):
	#	griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
	#griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy
	
	
	for i in range(20):
		trainer.trainEpochs( 1 )
		
		trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
		tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )
		print "epoch: %4d" % trainer.totalepochs, \
			  "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
		
	return
	
	
	#seems to use much more memory that running one epoch at a time... genreated memory errors.
	#trainer.trainUntilConvergence() # alternate to trainEpochs....	
	#trainer.trainUntilConvergence(maxEpochs=1, verbose=True, continueEpochs=10, validationProportion=0.25)	
		
	trnresult = percentError( trainer.testOnClassData(), trndata['class'] )
	tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'] )

	print "epoch: %4d" % trainer.totalepochs, \
			  "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
	
	return	
	
	out = fnn.activateOnDataset(griddata)
	out = out.argmax(axis=1)  # the highest output activation gives the class
	out = out.reshape(X.shape)
	
	figure(1)
		
		
		
		
	ioff()  # interactive graphics off
	clf()   # clear the plot
	hold(True) # overplot on
	for c in [0,1,2]:
		here, _ = where(tstdata['class']==c)
		plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
	if out.max()!=out.min():  # safety check against flat field
		contourf(X, Y, out)   # plot the contour
	ion()   # interactive graphics on
	draw()  # update the plot
	
if __name__=="__main__":
	run_stack(448)