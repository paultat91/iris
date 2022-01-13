#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:50:48 2021

@author: paul
"""

import torch 
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from datetime import datetime

   
today = datetime.now()
tic = time.perf_counter()

# Get cpu or gpu device for training.
#device = "cpu"                                            ## use this line for cpu
device = "cuda" if torch.cuda.is_available() else "cpu"  ## use this line for gpu

print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(4, 1, bias=True).double().to(device),
            nn.Tanh(),
            nn.Linear(1, 3, bias=True).double().to(device),
            #nn.Tanh(),
        )
        

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_tanh_stack(x)
        return logits


model = NeuralNetwork()
#print(model)

learning_rate = .5       # Step size in gradient descent
lessons = 2000          # Each lesson consists of the entire training set
students = 7776         #16807 #len(grid)   # Each student receives different initiailized weights
random_state = 56          # 56 splits with equal number of instances per class

iris_data = load_iris() # load the iris dataset

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# # Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=random_state)

#Standardization
for i in range(4):
    train_x[:,i] = (train_x[:,i] - np.mean(train_x[:,i]))/np.std(train_x[:,i])
    test_x[:,i] = (test_x[:,i] - np.mean(test_x[:,i]))/np.std(test_x[:,i])


t_train_x = torch.from_numpy(train_x).to(device)
t_train_y = torch.from_numpy(train_y).to(device)
t_test_x = torch.from_numpy(test_x).to(device)
t_test_y = torch.from_numpy(test_y).to(device)

initial_weights = np.genfromtxt(fname = 'initial_weights_torch.txt')
initial_weights = torch.from_numpy(initial_weights)
final_weights = torch.empty([students,13]).double()

for student in range(students):
    print(f"On student number: {student}")
    for name, param in model.named_parameters():
        #torch.nn.init.normal_(param, 0.,0.5)

        if name=='linear_tanh_stack.0.weight':
            torch.nn.init.constant_(param[0,0], initial_weights[student,0])
            torch.nn.init.constant_(param[0,1], initial_weights[student,1])
            torch.nn.init.constant_(param[0,2], initial_weights[student,2])
            torch.nn.init.constant_(param[0,3], initial_weights[student,3])
            
        if name=='linear_tanh_stack.0.bias':
            torch.nn.init.constant_(param, 0)
    
        if name=='linear_tanh_stack.2.weight':
            torch.nn.init.constant_(param[0,0], initial_weights[student,4])
            torch.nn.init.constant_(param[1,0], 0)
            torch.nn.init.constant_(param[2,0], initial_weights[student,6])
            
        if name=='linear_tanh_stack.2.bias':
            torch.nn.init.constant_(param[0], 0)
            torch.nn.init.constant_(param[1], 0)
            torch.nn.init.constant_(param[2], 0)
        
        #print(f"Layer: {name} | Size: {param.size()} | Values : {param[:]} \n")
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for lesson in range(lessons):
    
        logits = model(t_train_x)
        # pred_probab = nn.Softmax(dim=1)(logits)
        # y_pred = pred_probab.argmax(1) 
        # conf_matrix = confusion_matrix(t_train_y.argmax(1), y_pred)
        # print(conf_matrix)
        L = loss(logits, t_train_y.argmax(1))
        # Backpropagation
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        # if i % 100 == 0:
        #     L = L.item()
        #     print(f"training loss: {L:>7f}")
        for name, param in model.named_parameters():   
            if name=='linear_tanh_stack.2.weight':
                p=param.norm()
                torch.nn.init.constant_(param[0,0], 10*param[0,0]/p)
                torch.nn.init.constant_(param[1,0], 0)
                torch.nn.init.constant_(param[2,0], 10*param[2,0]/p)
            # if name=='linear_tanh_stack.2.bias':
            #     torch.nn.init.constant_(param[1], 0)

                    
    logits = model(t_test_x)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1) 
    L = loss(logits, t_test_y.argmax(1))
    
    #print(f"Predicted class: {y_pred} | Actual class: {Y.argmax(1)} | Loss: {L}")
    
    for name, param in model.named_parameters(): 
        if name=='linear_tanh_stack.0.weight':
            final_weights[student,0:4] = param  
          
        if name=='linear_tanh_stack.0.bias':
            final_weights[student,12] = param
    
        if name=='linear_tanh_stack.2.weight':
            final_weights[student,4:7] = param[:,0]  
              
        if name=='linear_tanh_stack.2.bias':
             final_weights[student,9:12] = param
            
        #print(f"Layer: {name} | Size: {param.size()} | Values : {param} \n")
    
    accuracy = (y_pred.size()[0] - torch.count_nonzero(y_pred - t_test_y.argmax(1)))/y_pred.size()[0]
    print(f"test loss: {L:>7f}, test acc: {accuracy}")
    #conf_matrix = confusion_matrix(t_test_y.argmax(1), y_pred)
    #print(conf_matrix)
    final_weights[student,7] = accuracy
    final_weights[student,8] = L

final = final_weights.detach().numpy()
total_lessons = 1*lessons
np.savetxt(f'final_weights_torch_{total_lessons}.txt', final)

toc = time.perf_counter()
time = toc - tic
print("time to complete: ", time)
