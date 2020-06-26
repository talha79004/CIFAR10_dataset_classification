#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# In[3]:


import import_ipynb
from get_2_class_new import*

#train_data2,train_label2,v_test,v_test_label2=getclass2(6,9)
train_data2,train_label2,v_test,v_test_label2=getclass5(0,1,3,6,9)


# In[4]:


from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#Convert numpy array(training) into tensor
x_train,y_train=map(torch.tensor,(train_data2,train_label2))
#print(x_train.dtype)
train_ds=TensorDataset(x_train,y_train)
train_dl=DataLoader(train_ds,batch_size=100)

#Convert numpy array(test) into tensor
x_test,y_test=map(torch.tensor,(v_test,v_test_label2))
test_ds=TensorDataset(x_test,y_test)
test_dl=DataLoader(test_ds,batch_size=100)




# In[5]:


import torch.nn as nn
import torch.nn.functional as F

#Defining network class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# In[6]:


import torch.optim as optim
#Defining Crossentropy loss and stochastic gradient descent
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[ ]:


co=[]
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    co.append(running_loss)
    print(running_loss)
fig = plt.figure()
plt.plot(co, color= "green")
plt.show()
print('Finished Training')


# In[ ]:


# Extracting test images and labels 
dataiter = iter(test_dl)
images, labels = dataiter.next()
outputs = net(images)

# Taking argmax of output predicted value
_, predicted = torch.max(outputs, 1)


# In[ ]:


#Finding the accuracy by comparing predicted  output with labels of test set
correct = 0
total = 0
#i=0
with torch.no_grad():
    for data in test_dl:
        #i=i+1
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
#print(i)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
print(images.shape)

