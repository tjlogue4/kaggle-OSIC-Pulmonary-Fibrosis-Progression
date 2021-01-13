import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import CSVLogger #this will save our training information to a log
from time import time
import pandas as pd
start_time = int(time())

#train = np.load('train.npy', allow_pickle = True)
#test = np.load('train.npy', allow_pickle = True)

training_data = np.load('data.npy', allow_pickle = True)

#print(len(training_data)) #total number of images
#print(len(training_data)*.8) #lets figure out roughly the 80% mark well train on 80% and test on 20%
#ill go with 25000
train_amount = 25000
test_amount = len(training_data) - train_amount

#print(f'Train# {train_amount}, Test {test_amount} Train% {train_amount/len(training_data)} Test% {test_amount/len(training_data)}')

#shuffle the data set before selecting
import random
random.shuffle(training_data)

train= training_data[:25000]
test = training_data[25000:]

X = []
y = []
for features, label in train:
        X.append(features)
        y.append(label)

Xtest = []
ytest = []
for features_test, label_test in test:
        Xtest.append(features_test)
        ytest.append(label_test)


X2 = np.array(X).reshape(-1,256,256,1) 
y2 = np.array(y)

Xtest = np.array(Xtest).reshape(-1,256,256,1) 
ytest = np.array(ytest)

input_shape = (256, 256, 1)

model_info_df = pd.DataFrame(columns = ['Time', 'dense_layer_size', 'dense_layers', 'layer_size', 'conv_layers', 'epoch', 'acc', 'loss', 'val_acc', 'val_loss', 'test_loss', 'test_acc'])
model_info_df.to_csv(f'model_log_train{start_time}.csv', index = False)

csv_logger = CSVLogger("model_history_log.csv", append=False)

#dense_layer_sizes = [8]#, 16, 32, 64]
#dense_layers = [1,2, 3]
#conv_layer_sizes = [8, 16, 32, 64]
#conv_layers = [1,2, 3]

dense_layer_sizes = [8]
dense_layers = [3]
conv_layer_sizes = [32]
conv_layers = [2]



for dense_layer_size in dense_layer_sizes:
	for dense_layer in dense_layers:
		for layer_size in conv_layer_sizes:
			for conv_layer in conv_layers:
				print(dense_layer_size, dense_layer, layer_size, conv_layer)

				model = Sequential()
				model.add(Conv2D(layer_size, (3,3), input_shape = input_shape))
				model.add(Activation("relu"))
				model.add(MaxPooling2D(pool_size = (2,2)))

				for l in range(conv_layer-1):

					model.add(Conv2D(layer_size, (3,3)))
					model.add(Activation("relu"))
					model.add(MaxPooling2D(pool_size = (2,2)))



				model.add(Flatten()) 

				for l in range(dense_layer):
					model.add(Dense(dense_layer_size))
					model.add(Activation("relu"))






				#output layer
				model.add(Dense(1)) 
				model.add(Activation('sigmoid')) 

				model.compile(loss = "binary_crossentropy",
				                         optimizer = 'adam',
				                         metrics = ['accuracy'])

				history = model.fit(X2, y2, batch_size =64, epochs =5, validation_split=.1)

				test_loss, test_acc =model.evaluate(Xtest, ytest, verbose=2)
				#print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

				for num in range(len(history.history['loss'])):
					
					model_info_df = pd.read_csv(f'model_log_train{start_time}.csv')
					data_to_add = {'Time': start_time, 'dense_layer_size': dense_layer_size, 'dense_layers':dense_layer, 'layer_size': layer_size, 'conv_layers': conv_layer, 'epoch': num, 'acc':history.history['accuracy'][num], 'loss':history.history['loss'][num], 'val_acc':history.history['val_accuracy'][num], 'val_loss':history.history['val_loss'][num], 'test_loss':test_loss, 'test_acc': test_acc}
					model_info_df = model_info_df.append(data_to_add, ignore_index = True)
					model_info_df.to_csv(f'model_log_train{start_time}.csv', index = False)


				#test_loss, test_accuracy =model.evaluate(Xtest, ytest, verbose=2)
				#print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


