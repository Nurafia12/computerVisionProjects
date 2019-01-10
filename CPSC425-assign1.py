# -*- coding: utf-8 -*-
#Import Statements
from PIL import Image
import numpy as np
import math
from scipy import signal



#Q1

# BoxFilter(n) takes in a dimension size n
# returns boxfilter of size n by n if n is odd
# returns Error if n is even
def boxfilter(n):
    assert (n % 2 != 0), 'Error: Filter needs to be odd dimensions'
    return np.ndarray(shape=(n,n), dtype=int, order='F')

print("\n BoxFilter of size 3:")    
print boxfilter(3)
print("\n BoxFilter of size 5:")  
print boxfilter(5)

#Q2

# gauss1d(sigma) prodcues a gaussian 1D filter for a given value of sigma
def gauss1d(sigma):
    s = int(math.ceil(sigma * 6))
    #if size is even change it to odd
    if(s % 2 == 0):
        s = s + 1
        
    #instatiate an array(filter) of size sigma * 6 and fill it in with random values
    filter = np.random.randint(10, size=s)
    
    #fill the values in the filter with the appropriate gaussian filter value
    filter = map(lambda y: np.exp(- y**2 / (2*sigma**2)), filter)
    
    #normalize the filter
    filter = map(lambda i: float(i)/sum(filter), filter)
    return filter

print("\n Gauss1d for sigma 0.3")        
print gauss1d(0.3)
print("\n Gauss1d for sigma 0.5")  
print gauss1d(0.5)
print("\n Gauss1d for sigma 1") 
print gauss1d(1)
print("\n Gauss1d for sigma 2") 
print gauss1d(2)

#Q3

# gauss2d(sigma) prodcues a gaussian 2D filter for a given value of sigma
def gauss2d(sigma):
    #Use gauss1d function to get 1D filter for a given sigma
    filter1D = np.array(gauss1d(sigma))
    #Convert the 1D filter to a 2D filter
    filter2D = filter1D[np.newaxis]
    #Convert the transpose of the 1D filter to a 2D transpose of the filter
    transpose2D = filter1D.T[np.newaxis]
    resultFilter = signal.convolve2d(filter2D, transpose2D, mode='full', boundary='fill')
    return resultFilter
    
print("\n Gauss2d for sigma 0.5")  
print gauss2d(0.5)
print("\n Gauss2d for sigma 1.0") 
print gauss2d(1.0)

#Q4a

# gaussconvolve2d(array,sigma) applies a 2D Gaussian filter to a given array
def gaussconvolve2d(array,sigma):
    filter = gauss2d(sigma)
    conv = signal.convolve2d(array,filter,'same')
    return conv

print("\n Gauss2dconvolution for sigma 0.5")      
print gaussconvolve2d([[1,2,3]],0.5)

#Q4a.2 Why does Scipy have separate functions ‘signal.convolve2d’ and ‘signal.correlate2d’?
# Signal.convole2d produces a image by applying the inverse of the filter whereas correlate 2d applies the filter


#Q4b

#Get the image of the dog
im = Image.open('/Users/pragyanbaidya/Downloads/dog.jpg')
#Convert image to grey scale
imGrey = im.convert('L')
#Convert image to numpy array with a Gauss2dConvolution of sigma 0.3
imArray = np.asarray(im.convert('L'))
print ("\n Numpy Array of the image of the dog")
convolvedArray = gaussconvolve2d(imArray,3)
imNew = Image.fromarray(convolvedArray)
print ("\n This is the convolved Array: ")
print (convolvedArray)

#Q4c
#show the original image
im.show()
#show the modified image
imNew.show()

#Q5
#We can use the seperatability priniciple of gaussian filter to perform two seprate convolution with two 1D gaussian filters instead of doing a single 2D convolution.
#We can then take the product of the two 1D convolutions to get the same result as that of the 2D gaussian convolved image. It would be much faster because 
#doing a single 2D convolution would required m^2 * n^2 multiplications whereas doing the product of two 1D convolutions would only require 2m* n^2 multiplications
#so therefore we save a lot of computation time.


#Part 2

#Q1

#Given an image and a sigma value produces a low frequency image 
def lowFreqImage(picture,sigma):
    im = Image.open(picture)
    imArray = np.asarray(im)
    #Seperate red,green and blue channels into seperate arrays
    redChannel = imArray[:,:,0]
    greenChannel = imArray[:,:,1]
    blueChannel = imArray[:,:,2]
    #Apply gauss filter on the color channels
    filteredRed = gaussconvolve2d(redChannel,sigma)
    filteredGreen = gaussconvolve2d(greenChannel,sigma)
    filteredBlue = gaussconvolve2d(blueChannel,sigma)
    #Put the filtered color channels into an image
    filteredImageArray = np.dstack((filteredRed,filteredGreen, filteredBlue))
    filteredImageArray = filteredImageArray.astype('uint8')
    imNew = Image.fromarray(filteredImageArray) 
    #Show the filtered image
    return imNew
    
lowFreqImage('/Users/pragyanbaidya/Downloads/dog.jpg',3).show()

#Q2
#Given an image and a sigma value produces a high frequency image
def highFreqImage(picture,sigma):
    im = Image.open(picture)
    imArray = np.asarray(im)
    #convert image array value to doubles
    imArray = imArray.astype(np.float)
    #Seperate red,green and blue channels into seperate arrays
    redChannel = imArray[:,:,0]
    greenChannel = imArray[:,:,1]
    blueChannel = imArray[:,:,2]
    #Apply gauss filter on the color channels
    filteredRed = gaussconvolve2d(redChannel,sigma)
    filteredGreen = gaussconvolve2d(greenChannel,sigma)
    filteredBlue = gaussconvolve2d(blueChannel,sigma)
    #Subtract the orginial rgb channels with the filtered rgb channels
    highFreqRed = np.subtract(redChannel,filteredRed)
    highFreqGreen = np.subtract(greenChannel,filteredGreen)
    highFreqBlue = np.subtract(blueChannel,filteredBlue)
    #Put the filtered color channels into an image
    filteredImageArray = np.dstack((highFreqRed,highFreqGreen, highFreqBlue))
    #Add 128 value to the image 
    pic = np.add(filteredImageArray,np.full(filteredImageArray.shape,128))
    arrayHigh = np.asarray(pic)
    arrayHigh = arrayHigh.astype('uint8')
    
    #Add 0.5 to the image
    for array in arrayHigh:     
       array = map(lambda x: x+0.5, array)
       
    #Convert the array to a high frequency image
    imNew = Image.fromarray(arrayHigh) 
    #Return the highfrequency image
    return imNew
    
highFreqImage('/Users/pragyanbaidya/Downloads/cat.jpg',1).show()

#Q3
def mergeHighAndLow(highImage,lowImage,sigmaHigh,sigmaLow):
    #Get the low frequency and high frequency images
    high = highFreqImage(highImage,sigmaHigh)
    highArray = np.asarray(high)
    low = lowFreqImage(lowImage,sigmaLow)
    lowArray = np.asarray(low)
    #convert image array value to doubles
    highArray = highArray.astype(np.float)
    lowArray = lowArray.astype(np.float)
    
    #Seperate the images into seperate color channels
    redChannelHigh = highArray[:,:,0]
    greenChannelHigh = highArray[:,:,1]
    blueChannelHigh = highArray[:,:,2]
    
    redChannelLow= lowArray[:,:,0]
    greenChannelLow = lowArray[:,:,1]
    blueChannelLow = lowArray[:,:,2]
    
    #Add the high and low color channels together
    combinedArrayRed = np.add(redChannelHigh,redChannelLow)
    combinedArrayGreen = np.add(greenChannelHigh,greenChannelLow)
    combinedArrayBlue = np.add(blueChannelHigh,blueChannelLow)
    
    #Put the filtered color channels into an image
    combinedArray = np.dstack((combinedArrayRed,combinedArrayGreen, combinedArrayBlue))
    combinedArray = combinedArray.astype('uint8')
    mergedImage = Image.fromarray(combinedArray)
    #Return the merged image
    return mergedImage

#Hybrid image with low image of sigma 0.2 and high image of sigma 3
mergeHighAndLow('/Users/pragyanbaidya/Downloads/cat.jpg','/Users/pragyanbaidya/Downloads/dog.jpg',3,3).show()

    
    
    
    




    
    


    
    
    
    
    
    
        
        
