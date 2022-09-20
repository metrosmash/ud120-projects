#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

from sklearn.svm import SVC
svm = SVC(kernel = 'sigmoid',C = 1,gamma = 'scale')


t0 = time()
svm.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t0 = time()
y_pred_svm = svm.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test,y_pred_svm)
print('The accuracy of the svm model is :',acc)
#sigmoid_kernel = 0.90 ,rbf_kernel = 0.88,poly_kernel = 0.61,linear_kernel = 0.884


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
