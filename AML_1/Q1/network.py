#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
class Network:

    def __init__(self):
        self.layers = []

# add layer to network
    def add(self, layer):
        self.layers.append(layer)
    
    
    def predict(self, input):
        # sample dimension first
        samples = len(input)
        result = []
  # forward propagation
        output = input
        for layer in self.layers:
            output = layer.forward(output)
            #result.append(output)
       # print(len(result))
        return output
    
    
    
    # train the network
    def fit(self, x_train, y_train,x_test,v_test_label2,  learning_rate,epochs=200):
        # sample dimension first
        samples = len(x_train)
        co=[]
        # training loop
        for i in range(epochs):        
        # forward propagation
            batch=0
            batch_size=50
            while batch < samples:
                output = x_train[batch:batch+batch_size,:]
                #print("xtrain shape",output.shape)
                #print('batch',batch)
                for layer in self.layers:
                    output = layer.forward(output)
                    #print("output",output.shape)



                exp_scores = np.exp(output-np.max(output, axis=1, keepdims=True))  
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                correct_logprobs = -np.log(probs[range(batch_size),y_train[batch:batch+batch_size]])
                correct_logprobs = np.array([correct_logprobs]).T
                data_loss = np.sum(correct_logprobs)/batch_size
                loss=data_loss
                

                dscores=probs
                dscores[range(batch_size),y_train[batch:batch+batch_size]]-=1
                dscores/=batch_size
                error=dscores
                

                # backward propagation
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                
               
                net=Network()
                out = net.predict(x_test)

                exp_scores = np.exp(out-np.max(out, axis=1, keepdims=True))  
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                predicted_class = np.argmax(probs, axis=1)
                
                #Updating batch size till all the samples are encountered for one epoch
                batch=batch+batch_size
            print ('training accuracy: %.2f' % (np.mean(predicted_class == v_test_label2)))
            print('loss of' ,i, 'is', loss)
            co.append(loss)

        fig = plt.figure()
        plt.plot(co, color= "green")
        plt.show()

