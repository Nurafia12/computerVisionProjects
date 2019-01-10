 #Starter code prepared by Borna Ghotbi for computer vision
 #based on MATLAB code by James Hay
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np

'''This function will predict the category for every test image by finding
the training image with most similar features. Instead of 1 nearest
neighbor, you can vote based on k nearest neighbors which will increase
performance (although you need to pick a reasonable value for k). '''

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Useful funtion:
    	
    	# You can use knn from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    '''
    
    k_nearest = 35
    #Create a classifier for k_nearest neighbors
    n = KNeighborsClassifier(n_neighbors=k_nearest)
    #Fit the model using train_image_feats as training data and train_labels as target values
    n.fit(train_image_feats,train_labels)
    
    #Classify each test feature into one of the training labels
    predicted_labels = n.predict(test_image_feats)
    len_pred_labels = len(predicted_labels)
    len_labels = len(set(train_labels))
    
    #Create the M*l 1 hot array where we can find which category each of the test image belongs 
    predicted_2d = np.zeros((len_pred_labels,len_labels))
    for index in range(len(predicted_labels)):
        pred_value = predicted_labels[index]
        #If our predicted label states that the test image belongs to a certain training label we make the correspoding value in our one hot-vector a 1 otherwise leave it as a 0
        for index_label in range(len_labels):
            if(index_label == pred_value):
                predicted_2d[index][index_label] = 1
         
    return predicted_2d
    

'''This function will train a linear SVM for every category (i.e. one vs all)
and then use the learned linear classifiers to predict the category of
very test image. Every test feature will be evaluated with all 15 SVMs
and the most confident SVM will "win". Confidence, or distance from the
margin, is W*X + B where '*' is the inner product or dot product and W and
B are the learned hyperplane parameters. '''

def svm_classify(train_image_feats, train_labels, test_image_feats):

    '''
    Parameters
        ----------
        train_image_feats:  is an N x d matrix, where d is the dimensionality of the feature representation.
        train_labels: is an N x l cell array, where each entry is a string 
        			  indicating the ground truth one-hot vector for each training image.
    	test_image_feats: is an M x d matrix, where d is the dimensionality of the
    					  feature representation. You can assume M = N unless you've modified the starter code.
        
    Returns
        -------
    	is an M x l cell array, where each row is a one-hot vector 
        indicating the predicted category for each test image.

    Useful funtion:
    	
    	# You can use svm from sci-kit learn.
        # Reference: https://scikit-learn.org/stable/modules/svm.html

    '''
    #Penalty parameter C of the error term. 
    c_val = 1000
    lin_svm = svm.LinearSVC(C= c_val)
    #Classify training image features into one of the training labels
    lin_svm.fit(train_image_feats, train_labels)
    #Use linear SVM to predict the class labels for each test image features
    predicted_labels = lin_svm.predict(test_image_feats)
    len_pred_labels = len(predicted_labels)
    len_labels = len(set(train_labels))
    
    #Initialize the M*l 1 hot array where we can find which category each of the test image belongs 
    predicted_2d = np.zeros((len_pred_labels,len_labels))
    for index in range(len(predicted_labels)):
        pred_value = predicted_labels[index]
        #If our predicted label states that the test image belongs to a certain training label we make the correspoding value in our one hot-vector a 1 otherwise leave it as a 0
        for index_label in range(len_labels):
            if(index_label == pred_value):
                predicted_2d[index][index_label] = 1
    
    #print "predicted labels for svm"
    #print predicted_2d
    return predicted_2d 

