#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np  
from util import sample_images, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


#--------------------------- Helper functions -----------------------------------------------------------
##Helper function to convert one hot vector for svm and knn to category names
def convertToCategoryNames(predicted_labels):
    #Build prediction_labels_names that contains predicition of category names for each test image. 
    #Create this by mapping each category number in prediction_labels to a corresponding category name 
    pred_labels_names = []
    for img in predicted_labels:
        for category,val in enumerate(img):
            if(val == 1):    
                pred_labels_names.append(category_labels[int(category)])
    return pred_labels_names
            

## Helper function to find accuracy for svm and knn
def find_accuracy(predicted_labels):   
    true_positive = 0
    #Build a confusion matrix that shows matrix of actual vs prediction        
    cm = confusion_matrix(test_labels_names, predicted_labels)
   
    #Get the diagonal of the confusion matrix that shows the number of true positives for our prediction
    diagonal = np.diag(cm)
    for i in range(len(diagonal)):
        #Add all the diagonal values to find total true positives
        true_positive = true_positive + diagonal[i]
    
    #Total accuracy is the total number of true positives divided the by the total test_labels
    acc = float(true_positive)/len(test_labels)
    return acc   
    
## Helper function to create confusion matrix for svm and knn    
def create_confusion_matrix(predicted_labels):
    #Building a confusion matrix   
    y_actu = pd.Series(test_labels_names, name='Actual')
    y_pred = pd.Series(predicted_labels, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    return df_confusion
    
#--------------------------- Helper functions -----------------------------------------------------------
    
    

#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = sample_images("/Users/pragyanbaidya/Downloads/hw5/sift/train", n_sample=300)
test_image_paths, test_labels = sample_images("/Users/pragyanbaidya/Downloads/hw5/sift/test", n_sample=100)
category_labels = ["Bedroom","Coast","Forest","Highway","Industrial","InsideCity","Kitchen","LivingRoom","Mountain","Office","OpenCountry","Store","Street","Suburb","TallBuilding"]
       

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

        
print('Extracting SIFT features\n')
#DONE: You code build_vocabulary function in util.py
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

#DONE: You code get_bags_of_sifts function in util.py 
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)


#Seperate all images into the categories they belong
dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[]}
for img_label,category_label in enumerate(train_labels):
    dict[int(category_label)].append(img_label)

#For each category of images compute the summation of their clusters and create a average histogram
for category in dict:
    imgs_in_category = dict[category]
    num_imgs_category = len(dict[category])
    vocab_size = len(train_image_feats[0])
    category_hist = [0] * vocab_size
    
    for img in imgs_in_category:
        for cluster_index,cluster_value in enumerate(train_image_feats[img]):
            category_hist[cluster_index] = category_hist[cluster_index] + cluster_value
   
    #Divide the cluster values by total # of images in category to compute the average histogram
    avg_category_hist = list(map(lambda x: x/vocab_size, category_hist))
    
    #Plot and display the histogram
    plt.bar(range(vocab_size), avg_category_hist, align='center', alpha=0.5)
    plt.ylabel('Number of freq')
    plt.xlabel('Vocab')
    plt.title(category_labels[category])
    #plt.show()
    plt.savefig(category_labels[category])
    plt.clf()

   
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.

''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')

#DONE: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
   
print('Using support vector machine to predict test set categories\n')

#DONE: YOU CODE svm_classify function from classifers.py
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)


'''Step 3: Build a confusion matrix and score the recognition system for 
         each of the classifiers.
DONE: In this step you will be doing evaluation. 
1) Calculate the total accuracy of your model by counting number
   of true positives and true negatives over all. (DONE)
2) Build a Confusion matrix and visualize it.    (DONE)
   You will need to convert the one-hot format labels back
   to their category name format. '''


#Convert our test_labels that contains category numbers to test_labels_names that contains the category names for each test image. 
test_labels_names = []
for val in test_labels:
    #Map each category number in test_labels list to a corresponding category name 
    test_labels_names.append(category_labels[int(val)])
    
#Convert one hot vector for our knn predictions into category name format
knn_names_labels = convertToCategoryNames(pred_labels_knn)
#Convert one hot vector for our svm predictions into category name format
svm_names_labels = convertToCategoryNames(pred_labels_svm)

#Find the accuracy for knn
knn_accuracy = find_accuracy(knn_names_labels)
print "This is knn_accuracy"
print knn_accuracy

#Find the accuracy for svm
svm_accuracy = find_accuracy(svm_names_labels)
print "This is svm accuracy"
print svm_accuracy

#Building a confusion matrix for knn
confusion_knn = create_confusion_matrix(knn_names_labels)
print "This is the confusion matrix for knn"
print confusion_knn

#Build a confusion matrix for svm
confusion_svm = create_confusion_matrix(svm_names_labels)
print "This is svm confusion matrix"
print confusion_svm


print('---Evaluation---\n')

# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. Can reach .60 with K-NN and
#                     different distance metrics.
#  accuracy ~= .60 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
#                     of clusters, SVM regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense SIFT features.
#  accuracy >= .80 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.