#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import import_ipynb



from hog_extract import*
# Importing function from file "get_classes_data" for 2 class and 5 class data
from get_classes_data import*

# Function calling for 2 class data(get2class) and 5 class(get5class)
#train_data2,v_test2,train_label2,v_test_label2=getclass2(6,9)
train_data2,v_test2,train_label2,v_test_label2=getclass5(0,1,3,6,9)


# In[2]:


(train),(test)= HOG(train_data2,v_test2)


# In[3]:


from FC import*
from activation import*
from network import Network


# In[ ]:


# training data : 10000/25000 samples
import numpy as np
x_train = train.astype('float32')
y_train = train_label2

# same for test data : 10000 samples
x_test = test.astype('float32')

#Defining Relu non-linearity and its derivative
def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)


#Creating Network architecture
net = Network()
net.add(FC(144, 100))                   # input_shape=(batch_size, 144)    ;   output_shape=(batch_size, 100)
net.add(ActivationLayer(ReLU, dReLU))   #Relu Non-liearity
net.add(FC(100, 10))                    # input_shape=(batch_size, 100)    ;   output_shape=(batch_size, 10)
net.add(ActivationLayer(ReLU, dReLU))   #Relu Non-liearity
net.add(FC(10, 5))                      # input_shape=(batch_size, 10)    ;   output_shape=(batch_size, 5)

#calling training and predict function within fit fucntion and plotting loss and accuracy for every epoch
net.fit(x_train, y_train, x_test, v_test_label2,  learning_rate=0.1,epochs=200)


# In[ ]:




