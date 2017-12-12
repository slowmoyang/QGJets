'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from iter import *
import argparse
import keras
from keras import backend as K
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.1
set_session(tf.Session(config=config))

batch_size = 100
num_classes = 2
epochs = 20

parser=argparse.ArgumentParser()
parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--epoch",type=int,default=10,help='epoch')
parser.add_argument("--load",type=str,default="weakdijet_0.8",help='load name')
parser.add_argument("--save",type=str,default=1,help='rch')
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

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
savename="save/"+args.load
model=keras.models.load_model(savename+"/_"+str(args.epoch))

#train=wkiter(["root/cutb/q"+str(int(args.rat*100))+"img.root","root/cutb/g"+str(int(args.rat*100))+"img.root"],batch_size=batch_size,end=1,istrain=1,friend=0)
test=wkiter("root",friend=20,begin=5./7.,end=1.,batch_size=batch_size)
from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score
x=[]
y=[]
g=[]
q=[]
entries=500
batch_num=batch_size
print ("eval",test.totalnum())
for j in range(entries):
	a,c=test.test()
	b=model.predict(a,verbose=0)[:,0]
	x=np.append(x,np.array(c[:,0]))
	y=np.append(y,b)
	for i in range(batch_num):
		if(c[i][0]==1):
			g.append(b[i])
		else:
			q.append(b[i])
plt.figure(1)
plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.7,label='quark')
plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.7,label='gluon')
plt.legend(loc="upper center")
plt.savefig(savename+"/like"+str(args.epoch)+".png")
t_fpr,t_tpr,_=roc_curve(x,y)
t_fnr=1-t_fpr
train_auc=np.around(auc(t_fpr,t_tpr),4)
plt.figure(2)
plt.plot(t_tpr,t_fnr,alpha=0.5,label="AUC={}".format(train_auc),lw=2)
plt.legend(loc='lower left')
plt.savefig(savename+"/roc"+str(args.epoch)+".png")
#print(b,c)

