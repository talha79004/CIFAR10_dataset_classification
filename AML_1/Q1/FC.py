#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
class FC:
    def __init__(self, input_size, output_size):
        self.weights = 0.01*np.random.randn(input_size, output_size)
        self.bias = np.zeros((1,output_size))
        
    def forward(self,input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
       # print("outerror",output_error.shape)
      #  print("wt shape",self.weights.shape)
        
        weights_error = np.dot(self.input.T, output_error)
       # print("wt eroor",weights_error.shape)
        dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error 
      #  print("upd wt",self.weights.shape)
        self.bias -= learning_rate * np.sum(output_error,axis=0,keepdims=True)
        #print("upd bias",self.bias.shape)
        return input_error

