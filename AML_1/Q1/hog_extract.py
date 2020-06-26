#!/usr/bin/env python
# coding: utf-8

# In[3]:


def HOG(train_data2,v_test2):
    import numpy as np
    from skimage import feature
    # Extracting HOG features from train data set
    train=[]
    for image in train_data2:
        H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=False,multichannel=True)
        train.append(H)
    train=np.array(train)
    print("train shape after HOG",train.shape)

    # Extracting HOG features from train data set
    test=[]
    for image in v_test2:
        H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=False,multichannel=True)
        test.append(H)
    test=np.array(test)
    print("test shape after HOG",test.shape)
    #print(test[0,:])
    return train,test

