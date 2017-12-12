'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module

import os
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4

set_session(tf.Session(config=config))

batch_size = 500
num_classes = 2
epochs = 20

parser=argparse.ArgumentParser()

parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--save",type=str,default="weakdijet_",help='save name')

args=parser.parse_args()

# input image dimensions
img_rows, img_cols = 33, 33

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (3,img_rows, img_cols)

net=import_module('symbols.'+"asvgg")
model=net.get_symbol(input_shape,num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

#ab=int(len(x_train)/10)

train=wkiter(["root/cutb/q"+str(int(args.rat*100))+"img.root","root/cutb/g"+str(int(args.rat*100))+"img.root"],batch_size=batch_size,end=0.01,istrain=1,friend=0)
test=wkiter("root",friend=20,begin=5./7.,end=5.1/7.,batch_size=batch_size)
savename='save/'+str(args.save)+str(args.rat)
os.system("mkdir "+savename)
print ("train",train.totalnum(),"eval",test.totalnum())
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=batch_size)

for i in range(epochs):
	print("epoch",i)
	checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/_'+str(i),monitor='val_loss',verbose=0,save_best_only=False,mode='auto')
	model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),validation_data=test.next(),validation_steps=test.totalnum(),epochs=1,verbose=1,callbacks=[logger,checkpoint])

	train.reset()
	test.reset()
	"""while True:
		X_train,Y_train=train.next()
		#X_train=x_train[j*ab:(j+1)*ab]
		#Y_train=y_train[j*ab:(j+1)*ab]
		#print(type(X_train),type(Y_train.shape))
		#print(X_train.shape,Y_train.shape)
		if(train.endfile==1):
			model.fit(X_train, Y_train,
				epochs=1,
				verbose=1,
				validation_data=(test.next())
				)
			test.reset()
			train.reset()
			break
		else:
			model.fit(X_train, Y_train,
				epochs=1,
				verbose=0,
				)"""
		


