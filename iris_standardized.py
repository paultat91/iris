#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:37:53 2020

@author: paul
"""


import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad 
import autograd.numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def relu(x):
    #return np.max(0,x)
    return x * (x > 0)
    #return np.tanh(x)

def softmax(x):
    exp = np.exp(x) 
    return exp / exp.sum(0)

def Error(W_hh, W_oh, b_h, b_o, x, y):
    h = relu(np.dot(W_hh,x.T) + b_h)
    y_hat = softmax(np.dot(W_oh,h) + b_o)
    return np.sum(np.diag(np.dot(y, -np.log(y_hat))))/len(x)

def forward(x, y, W_hh, W_oh, b_h, b_o):
    h = relu(np.dot(W_hh,x.T) + b_h)
    y_hat = softmax(np.dot(W_oh,h) + b_o)
    pred = np.expand_dims(np.argmax(y_hat, axis=0), axis=0).T
    num_wrong = np.count_nonzero(encoder.inverse_transform(y) - pred)
    acc = (len(x) - num_wrong)/len(x)
    err = Error(W_hh, W_oh, b_h, b_o, x, y) 
    return acc, err

def update(W_hh, W_oh, b_h, b_o, x, y):
    dE_dWhh = grad(Error, argnum=0)(W_hh, W_oh, b_h, b_o, x, y)
    dE_dWoh = grad(Error, argnum=1)(W_hh, W_oh, b_h, b_o, x, y)
    dE_dbh = grad(Error, argnum=2)(W_hh, W_oh, b_h, b_o, x, y)
    dE_dbo = grad(Error, argnum=3)(W_hh, W_oh, b_h, b_o, x, y)
    
    W_hh = W_hh - learning_rate*dE_dWhh
    W_oh = W_oh - learning_rate*dE_dWoh
    b_h = b_h - learning_rate*dE_dbh
    b_o = b_o - learning_rate*dE_dbo
    
    #W_oh[1]=0 # freeze 2nd W_oh to zero
    
    # if W_oh[0]>boundary:
    #     W_oh[0] = boundary
    # if W_oh[0]<-boundary:
    #     W_oh[0] = -boundary
        
    # if W_oh[2]>boundary:
    #     W_oh[2] = boundary
    # if W_oh[2]<-boundary:
    #     W_oh[2] = -boundary 
        
    return W_hh, W_oh, b_h, b_o, dE_dWhh, dE_dWoh, dE_dbh, dE_dbo


n = 1                   # Number of hidden units? layers?
learning_rate = 0.1     # Step size in gradient descent
boundary = 5

students = 5         # Each student receives different itiailized weights
lessons = 20000          # Each lesson consists of the entire training set
   
iris_data = load_iris() # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# # Split the data for training and testing
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

initial_weights = np.genfromtxt(fname='inputs/SF5d_5.dat')
#initial_weights = np.genfromtxt(fname='inputs/keras_inputs.txt')

train_x = np.genfromtxt(fname='same_split/train_x.txt')
train_y = np.genfromtxt(fname='same_split/train_y.txt')
test_x = np.genfromtxt(fname='same_split/test_x.txt')
test_y = np.genfromtxt(fname='same_split/test_y.txt')

train_x = np.delete(np.delete(train_x, 0 , 1), 0, 1)
test_x = np.delete(np.delete(test_x, 0 , 1), 0, 1)
x = np.delete(np.delete(x, 0 , 1), 0, 1)

#Standardization
train_x[:,0] = (train_x[:,0] - np.mean(train_x[:,0]))/np.std(train_x[:,0])
train_x[:,1] = (train_x[:,1] - np.mean(train_x[:,1]))/np.std(train_x[:,1])
test_x[:,0] = (test_x[:,0] - np.mean(test_x[:,0]))/np.std(test_x[:,0])
test_x[:,1] = (test_x[:,1] - np.mean(test_x[:,1]))/np.std(test_x[:,1])


for out in range(1):

    acc = np.array([])
    err = np.array([])
    
    Whh1 = np.array([])
    Whh2 = np.array([])
  
    Woh1 = np.array([])
    Woh2 = np.array([])
    Woh3 = np.array([])
     
    bh1 = np.array([])
    bo1 = np.array([])
    bo2 = np.array([])
    bo3 = np.array([])
    
    dWhh1 = np.array([])
    dWhh2 = np.array([])
 
    dWoh1 = np.array([])
    dWoh2 = np.array([])
    dWoh3 = np.array([])
    
    dWhh = np.array([])
    dWoh = np.array([])
    dbh = np.array([])
    dbo = np.array([])
     
    output = np.array([[]])
    
    for student in range(students):
    
        W_hh = np.expand_dims(initial_weights[student,0:2], axis=1).T
        W_oh = np.expand_dims(initial_weights[student,2:5], axis=1)
               
        ## Initialization of bias parameters
        b_h = np.zeros([n,1])
        b_o = np.zeros([3,1])
        
        for lesson in range(lessons):
            W_hh, W_oh, b_h, b_o, dE_dWhh, dE_dWoh, dE_dbh, dE_dbo = update(W_hh, W_oh, b_h, b_o, train_x, train_y)
            Woh1 = np.append(Woh1, W_oh[0])
            Woh2 = np.append(Woh2, W_oh[1])
            Woh3 = np.append(Woh3, W_oh[2])

        test_acc, test_err = forward(test_x, test_y, W_hh, W_oh, b_h, b_o)
        acc = np.append(acc, test_acc)
        err = np.append(err, test_err)
        
        o = np.array([])
        o = np.append(W_hh, W_oh)
        o = np.append(o, acc[student])
        o = np.append(o, err[student])
        o = np.append(o, b_o)
        o = np.append(o, b_h)
        o = np.expand_dims(o, axis=0)
        if student==0:
            output = np.append(output, o, axis=1)
        else:
            output = np.append(output, o, axis=0)
        plt.figure() 
        
        plt.plot(Woh1[-lessons:], label='Wo1') 
        plt.plot(Woh2[-lessons:], label='Wo2')
        plt.plot(Woh3[-lessons:], label='Wo3')
        plt.legend()
        plt.xlabel('Lessons (updates)')
        plt.ylabel('Wo')
        plt.title('Student number = ' + str(student) + str(',  ') + 'accuracy = ' + str(np.round(acc[student], 2)) + str(',  ')+str('Wh not fixed'))
        plt.show()
    
    #np.savetxt('conv_movie/output'+str(lessons)+'_relu_cutoffWoh_at5_standardized.txt', output)
    #lessons += 10

## PLOTS
color='copper'
Wh1 = output[:,0]
Wh2 = output[:,1]

Wo1 = output[:,2]
Wo2 = output[:,3]
Wo3 = output[:,4]

acc = output[:,5]
err = output[:,6]

bo1 = output[:,7]
bo2 = output[:,8]
bo3 = output[:,9]

bh = output[:,10]
# plt.figure() 
# plt.scatter(Wh1,Wh2, c=acc, cmap=color) 
# plt.xlabel('Wh1')
# plt.ylabel('Wh2')
# # axes = plt.gca()
# # axes.set_ylim([-2,4])
# # axes.set_xlim([-2,2])
# plt.title('Number of Updates = ' + str(lessons))
# plt.show()

# plt.figure() 
# plt.scatter(Wo1,Wo3, c=acc, cmap=color) 
# plt.xlabel('Wo1')
# plt.ylabel('Wo3')
# plt.title('Number of Updates = ' + str(lessons))
# plt.show()

# plt.figure()
# acc_plot = plt.scatter(err,acc)    
# plt.xlabel('err')
# plt.ylabel('acc')
# axes = plt.gca()
# axes.set_ylim([0,1])
# axes.set_xlim([0,8])
# plt.title('Number of Updates = ' + str(lessons))
# plt.show()


