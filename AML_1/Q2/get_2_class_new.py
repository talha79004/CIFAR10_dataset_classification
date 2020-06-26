#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Function for getting data of 2 classes
def getclass2(class1,class2):
    import numpy as np
    import pandas as pd
    from six.moves import cPickle as pickle
    import matplotlib.pyplot as plt
    
    #Loading each batch in dictionary
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
# Extracting data and labels from each batch(1 to 5 and test batch) 
    b1=unpickle('data_batch_1')
    b1_data= b1[b'data']
    b1_data=np.array(b1_data)
    b1_labels= b1[b'labels']
    b1_labels=np.array([b1_labels]).T


    b2=unpickle('data_batch_2')
    b2_data= b2[b'data']
    b2_data=np.array(b2_data)
    b2_labels= b2[b'labels']
    b2_labels=np.array([b2_labels]).T


    b3=unpickle('data_batch_3')
    b3_data= b3[b'data']
    b3_data=np.array(b3_data)
    b3_labels= b3[b'labels']
    b3_labels=np.array([b3_labels]).T

    
    b4=unpickle('data_batch_4')
    b4_data= b4[b'data']
    b4_data=np.array(b4_data)
    b4_labels= b4[b'labels']
    b4_labels=np.array([b4_labels]).T
    

    b5=unpickle('data_batch_5')
    b5_data= b5[b'data']
    b5_data=np.array(b5_data)
    b5_labels= b5[b'labels']
    b5_labels=np.array([b5_labels]).T
    

    b_test=unpickle('test_batch')
    b_test_data= b_test[b'data']
    b_test_data=np.array(b_test_data)
    b_test_labels= b_test[b'labels']
    b_test_labels=np.array([b_test_labels]).T
    
# Obtaining indices for class labels to extract data corresponding to those classes from batches
    ro1=[]
    for i in range(len(b1_labels)):
        if b1_labels[i] == class1 or b1_labels[i]==class2:
            ro1.append(i)
    v=np.array(b1_data[ro1])
    v_label=np.array(b1_labels[ro1])


    ro2=[]
    for i in range(len(b2_labels)):
        if b2_labels[i] == class1 or b2_labels[i]==class2:
            ro2.append(i)
    v2=np.array(b1_data[ro2])
    v2_label=np.array(b2_labels[ro2])


    ro3=[]
    for i in range(len(b3_labels)):
        if b3_labels[i] == class1 or b3_labels[i]==class2:
            ro3.append(i)
    v3=np.array(b3_data[ro3])
    v3_label=np.array(b3_labels[ro3])

    
    ro4=[]
    for i in range(len(b4_labels)):
        if b4_labels[i] == class1 or b4_labels[i]==class2:
            ro4.append(i)
    v4=np.array(b4_data[ro4])
    v4_label=np.array(b4_labels[ro4])
    
    
    ro5=[]
    for i in range(len(b5_labels)):
        if b5_labels[i] == class1 or b5_labels[i]==class2:
            ro5.append(i)
    v5=np.array(b5_data[ro5])
    v5_label=np.array(b5_labels[ro5])


    ro_test=[]
    for i in range(len(b_test_labels)):
        if b_test_labels[i] == class1 or b_test_labels[i]==class2:
            ro_test.append(i)
    v_test=np.array(b_test_data[ro_test])
    v_test_label=np.array(b_test_labels[ro_test])


# Reshaping test data in the form of 32x32x3 using reshape and rollaxis
    l=len(v_test)
    v_test=np.array(v_test).reshape(l,3,32,32)
    v_test=np.float32(v_test)
    #print("v_test data shap",v_test.shape)


# Making label of first class1=0 and class2=1 for test data
    v_test_label=np.int64(v_test_label)
    v_test_label2=[0 if x==class1 else 1 for x in v_test_label]
    v_test_label2=np.array(v_test_label2)
    v_test_label2=np.int64(v_test_label2)
    #print("teset shap",v_test_label2.dtype)


# Concatenating all the data obtained from batches in one array
    train_data=np.concatenate((v, v2,v3,v4,v5), axis=0, out=None)
    train_label=np.concatenate((v_label,v2_label,v3_label,v4_label,v5_label),axis=0,out=None)
    train_data=np.float32(train_data)
    train_label=np.int64(train_label)
#     print(train_data.shape)
#     #print(train_label[0:10])
#     print(train_data.dtype)
#     print(train_label.dtype)

# Reshaping train data in the form of 3x32x32 using reshape and rollaxis 
    l=len(train_data)
    train_data2=np.array(train_data).reshape(l,3,32,32)
#     print(train_data2.shape)
#     print(train_data2.dtype)

# Making label of first class1=0 and class2=1 for test data
    train_label2=[0 if x==class1 else 1 for x in train_label]
    train_label2=np.array(train_label2)
    train_label2=np.int64(train_label2)
#   print("t",train_label2.dtype)
    
    return train_data2,train_label2,v_test,v_test_label2


# In[2]:


# Function for getting data of 2 classes
def getclass5(class1,class2,class3,class4,class5):
    import numpy as np
    import pandas as pd
    from six.moves import cPickle as pickle
    import matplotlib.pyplot as plt
    
    #Loading each batch in dictionary
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
# Extracting data and labels from each batch(1 to 5 and test batch) 
    b1=unpickle('data_batch_1')
    b1_data= b1[b'data']
    b1_data=np.array(b1_data)
    b1_labels= b1[b'labels']
    b1_labels=np.array([b1_labels]).T


    b2=unpickle('data_batch_2')
    b2_data= b2[b'data']
    b2_data=np.array(b2_data)
    b2_labels= b2[b'labels']
    b2_labels=np.array([b2_labels]).T


    b3=unpickle('data_batch_3')
    b3_data= b3[b'data']
    b3_data=np.array(b3_data)
    b3_labels= b3[b'labels']
    b3_labels=np.array([b3_labels]).T


    b4=unpickle('data_batch_4')
    b4_data= b4[b'data']
    b4_data=np.array(b4_data)
    b4_labels= b4[b'labels']
    b4_labels=np.array([b4_labels]).T


    b5=unpickle('data_batch_5')
    b5_data= b5[b'data']
    b5_data=np.array(b5_data)
    b5_labels= b5[b'labels']
    b5_labels=np.array([b5_labels]).T


    b_test=unpickle('test_batch')
    b_test_data= b_test[b'data']
    b_test_data=np.array(b_test_data)
    b_test_labels= b_test[b'labels']
    b_test_labels=np.array([b_test_labels]).T

# Obtaining indices for class labels to extract data corresponding to those classes from batches
    ro1=[]
    for i in range(len(b1_labels)):
        if b1_labels[i] == class1 or b1_labels[i]==class2 or b1_labels[i] == class3 or b1_labels[i]==class4 or b1_labels[i] == class5:
            ro1.append(i)
    v=np.array(b1_data[ro1])
    v_label=np.array(b1_labels[ro1])


    ro2=[]
    for i in range(len(b2_labels)):
        if b2_labels[i] == class1 or b2_labels[i]==class2 or b2_labels[i]==class3 or b2_labels[i]==class4 or b2_labels[i]==class5:
            ro2.append(i)
    v2=np.array(b1_data[ro2])
    v2_label=np.array(b2_labels[ro2])


    ro3=[]
    for i in range(len(b3_labels)):
        if b3_labels[i] == class1 or b3_labels[i]==class2 or b3_labels[i]==class3 or b3_labels[i]==class4 or b3_labels[i]==class5:
            ro3.append(i)
    v3=np.array(b3_data[ro3])
    v3_label=np.array(b3_labels[ro3])


    ro4=[]
    for i in range(len(b4_labels)):
        if b4_labels[i] == class1 or b4_labels[i]==class2 or b4_labels[i]==class3 or b4_labels[i]==class4 or b4_labels[i]==class5:
            ro4.append(i)
    v4=np.array(b4_data[ro4])
    v4_label=np.array(b4_labels[ro4])


    ro5=[]
    for i in range(len(b5_labels)):
        if b5_labels[i] == class1 or b5_labels[i]==class2 or b5_labels[i]==class3 or b5_labels[i]==class4 or b5_labels[i]==class5:
            ro5.append(i)
    v5=np.array(b5_data[ro5])
    v5_label=np.array(b5_labels[ro5])


    ro_test=[]
    for i in range(len(b_test_labels)):
        if b_test_labels[i] == class1 or b_test_labels[i]==class2 or b_test_labels[i]==class3 or b_test_labels[i]==class4 or b_test_labels[i]==class5:
            ro_test.append(i)
    v_test=np.array(b_test_data[ro_test])
    v_test_label=np.array(b_test_labels[ro_test])

# Reshaping test data in the form of 32x32x3 using reshape and rollaxis
    l=len(v_test)
    v_test=np.array(v_test).reshape(l,3,32,32)
    v_test=np.float32(v_test)
    print("v_test data shap",v_test.shape)


# Making label of first class1 = 0, class2 = 1, class3 = 2, class4 = 3 and class5 = 4 for test data
    v_test_label=np.int64(v_test_label)
    v_test_label2=[0 if x==class1 else 1 if x==class2 else 2 if x==class3 else 3 if x==class4 else 4 for x in v_test_label]
    v_test_label2=np.array(v_test_label2)
    v_test_label2=np.int64(v_test_label2)
#    print("teset shap",v_test_label2.dtype)

# Concatenating all the data obtained from batches in one array
    train_data=np.concatenate((v, v2,v3,v4,v5), axis=0, out=None)
    train_label=np.concatenate((v_label,v2_label,v3_label,v4_label,v5_label),axis=0,out=None)
    train_data=np.float32(train_data)
    train_label=np.int64(train_label)
#     print(train_data.shape)
#     #print(train_label[0:10])
#     print(train_data.dtype)
#     print(train_label.dtype)

# Reshaping test data in the form of 32x32x3 using reshape and rollaxis
    l=len(train_data)
    train_data2=np.array(train_data).reshape(l,3,32,32)
#     print(train_data2.shape)
#     print(train_data2.dtype)

    
# Making label of first class1 = 0, class2 = 1, class3 = 2, class4 = 3 and class5 = 4 for test data
    train_label2=[0 if x==class1 else 1 if x==class2 else 2 if x==class3 else 3 if x==class4 else 4 for x in train_label]
    train_label2=np.array(train_label2)
    train_label2=np.int64(train_label2)
    print("t",train_label2.dtype)
    
    return train_data2,train_label2,v_test,v_test_label2


# In[ ]:




