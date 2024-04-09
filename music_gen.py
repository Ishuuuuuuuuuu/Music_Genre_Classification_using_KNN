#!/usr/bin/env python
# coding: utf-8

# In[14]:


from python_speech_features import mfcc     
import scipy.io.wavfile as wav              #extract and read the wave files
import numpy as np

from tempfile import TemporaryFile                                         #   (prevents data loss and free memory)

import os   
import pickle   #serializing and non serializing python objects
import random   
import operator   

import math

import librosa              #music and audio analysis (building blocks necessary to create music information retrieval systems.)
import librosa.display
from IPython.display import Audio              


# In[15]:


# function to get the distance between feature vectors and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors


# In[16]:


# identify the class of the instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]                                                                    #3 were of class a and 4 were of class b
        if response in classVote:
            classVote[response] += 1                                                           
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True) #returns a callable object that fetches item from it's operand

    return sorter[0][0]


# In[17]:


def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):                                                        #all those which are classified correctly are divided by it's length
        if testSet[x][-1] == predictions[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)


# In[18]:


# directory that holds the wav files
directory = "C:/Users/DELL/Desktop/MINI3(MUSIC2)/DATA/genres_original/"


f = open("my.dat", 'wb')      #dat file contains binary text

i = 0

for folder in os.listdir(directory):       #to list files and directories in a specified directories
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):        
        try:
            (rate, sig) = wav.read(directory+folder+"/"+file)                            #captures all the feature that we need...
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)                                                                                #serialize
        except Exception as e:
            print('Got an exception: ', e, ' in folder: ', folder, ' filename: ', file)        

f.close()


# In[19]:


# we will Split the dataset into training and testing sets respectively
dataset = []

def loadDataset(filename, split, trSet, teSet):
    with open('my.dat', 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))                                                                     #deserialize
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])
trainingSet = []
testSet = []
loadDataset('my.dat', 0.66, trainingSet, testSet)


# In[20]:


def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance


# In[21]:


# making predictions using KNN
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)


# In[22]:


from collections import defaultdict
results = defaultdict(int)                                                                   #dictionary for folder names
                                                          #key-value pair so that no display in int but the genre names
i=1
for folder in os.listdir(directory):
    results[i] = folder
    i+=1

print(results)


# In[38]:


# testing the code with test samples

#test_dir = "C:/Users/DELL/Desktop/MINI3(MUSIC2)/test/"
#test_dir = "C:/Users/DELL/Desktop/MINI3(MUSIC2)/DATA/genres_original/Blues/"
test_dir = "C:/Users/DELL/Desktop/MINI3(MUSIC2)/DATA/genres_original/Metal/"
#test_file = test_dir + "test5.wav"                 
#test_file = test_dir + "test2.wav"     
#test_file = test_dir + "test6.wav"                    
#test_file = test_dir + "blues.00002.wav"             
test_file = test_dir + "metal.00020.wav"    


# In[39]:


data1 , sr = librosa.load(test_file)              #RETURNS audio time series


# In[35]:


librosa.load(test_file, sr=45600)               #Load an audio file as a floating point time series (as per the given rate)


# In[36]:


import IPython
IPython.display.Audio(data1, rate=sr)                                                                            #to play audio


# In[31]:


(rate, sig) = wav.read(test_file)
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, i)


# In[32]:


pred = nearestClass(getNeighbors(dataset, feature, 5))
print(results[pred])


# In[ ]:




