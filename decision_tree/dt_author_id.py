#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dtc = DecisionTreeClassifier(max_depth = 150,min_samples_split= 300,criterion='entropy')

t1 = time()
dtc.fit(features_train,labels_train)
print("Training Time:", round(time()-t1, 3), "s")

t1 =  time()
y_pred_dtc = dtc.predict(features_test)
print('Predicting time:',round(time()-t1, 3), 's')

acc_dtc = accuracy_score(labels_test,y_pred_dtc)
print('the accuracy of the decision tree is :', acc_dtc)



#########################################################


