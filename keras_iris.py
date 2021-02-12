#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:37:40 2020

@author: paul
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD

iris_data = load_iris() # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
#print(y)

# Split the data for training and testing
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

train_x = np.genfromtxt(fname='same_split/train_x.txt')
train_y = np.genfromtxt(fname='same_split/train_y.txt')
test_x = np.genfromtxt(fname='same_split/test_x.txt')
test_y = np.genfromtxt(fname='same_split/test_y.txt')
train_x = np.delete(np.delete(train_x, 0 , 1), 0, 1)
test_x = np.delete(np.delete(test_x, 0 , 1), 0, 1)

#Standardization
train_x[:,0] = (train_x[:,0] - np.mean(train_x[:,0]))/np.std(train_x[:,0])
train_x[:,1] = (train_x[:,1] - np.mean(train_x[:,1]))/np.std(train_x[:,1])
test_x[:,0] = (test_x[:,0] - np.mean(test_x[:,0]))/np.std(test_x[:,0])
test_x[:,1] = (test_x[:,1] - np.mean(test_x[:,1]))/np.std(test_x[:,1])


inputs_Wh = np.empty((1,2))
outputs_Wh = np.empty((1,2))

inputs_Wo = np.empty((3,1))
outputs_Wo = np.empty((3,1))

acc = np.array([])
err = np.array([])

students = 100
lessons = 1000

for x in range(students):
    # Build the model
    model = Sequential()    
    model.add(Dense(1, input_shape=(2,), activation='relu', name='fc1',use_bias=True))
    model.add(Dense(3, activation='softmax', name='output',use_bias=True))
    
    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.1)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    input_weights = model.get_weights()
    inputs_Wh = np.append(inputs_Wh, input_weights[0].T, axis=0)
    inputs_Wo = np.append(inputs_Wo, input_weights[2].T, axis=1)
    
    # Train the model
    model.fit(train_x, train_y, verbose=0, batch_size=50, epochs=lessons)
    
    # Test on unseen data    
    results = model.evaluate(test_x, test_y, verbose=0)
    output_weights = model.get_weights()
    outputs_Wh = np.append(outputs_Wh, output_weights[0].T, axis=0)    
    outputs_Wo = np.append(outputs_Wo, output_weights[2].T, axis=1)    
    
    acc = np.append(acc, results[1])
    err = np.append(err, results[0])

    print(x, results[1])
    
inputs_Wh = np.delete(inputs_Wh, obj=0, axis=0)
outputs_Wh = np.delete(outputs_Wh, obj=0, axis=0)
inputs_Wo = np.delete(inputs_Wo, obj=0, axis=1)
outputs_Wo = np.delete(outputs_Wo, obj=0, axis=1)

plt.figure()
plt.scatter(inputs_Wh[:,0],inputs_Wh[:,1], c='r') 
plt.scatter(outputs_Wh[:,0],outputs_Wh[:,1], c='b') 
plt.xlabel('Wh1')
plt.ylabel('Wh2')
plt.title('Kernel Weights')
plt.show()

plt.figure()
plt.scatter(inputs_Wo[0,:],inputs_Wo[2,:], c='r') 
plt.scatter(outputs_Wo[0,:],outputs_Wo[2,:], c='b') 
plt.xlabel('Wo1')
plt.ylabel('Wo3')
plt.title('Output Weights')
plt.show()

plt.figure()
acc_plot = plt.scatter(err,acc)    
plt.xlabel('err')
plt.ylabel('acc')
axes = plt.gca()
axes.set_ylim([0,1])
axes.set_xlim([0,8])
plt.title('acc')
plt.show()

acc_thresh = .9
count=0
for i in range(len(acc)):
    if acc[i]>acc_thresh:
        count+=1
print(count)

#model.summary()
inputs_keras = np.append(inputs_Wh, inputs_Wo.T, axis=1)
outputs_keras = np.append(outputs_Wh, outputs_Wo.T, axis=1)

np.savetxt('inputs/keras_inputs_1000epochs_100iterations_batch50_Adam.txt', inputs_keras)
np.savetxt('outputs/keras_outputs_1000epochs_100iterations_batch50_Adam.txt', outputs_keras)

