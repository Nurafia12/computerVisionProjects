from PIL import Image, ImageDraw
import numpy as np
import math
from scipy import signal
import ncc


#------------------------------START OF ASSIGNMENT 2----------------------------------------------------------

#Open images
sportsImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/sports.jpg")
judyBatsImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/judybats.jpg")
templateImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/template.jpg")
fansImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/fans.jpg")
studentsImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/students.jpg")
familyImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/family.jpg")
treeImage = Image.open("/Users/pragyanbaidya/Downloads/hw2/faces/tree.jpg")


#Q2
# Image -> pyramidList
# Takes in a image and makes a Gaussian Pyramid for the given image where the smallest image dimension (i.e width or heigth) > minSize
# It returns the Gaussian Pyramid as a list of image of different dimensions

def makePyramid(image,minSize):
    pyramidList = [image];
    #keep reducing the size of the image and adding into the pyramid till height or width > minSize
    while (image.size[0]*0.75 > minSize or image.size[1]*0.75 > minSize):
        y = image.size[1]
        x = image.size[0]
        #Reduce each subsequent image in the pyramid by factor of 0.75
        image = image.resize((int(x*0.75),int(y*0.75)), Image.BICUBIC)
        pyramidList.append(image)       
    return pyramidList


#Q3
# PyramidList -> Void
# Takes a list of Image (Guassian pyramid) and combines them horizontally before printing it
def showPyramid(pyramidList):
    maxHeight = 0
    sumWidth = 0
    #Find the dimensions for the empty image frame
    for im in pyramidList:
        #add all the horizontal dimensions together to make the width of the frame
        x = im.size[0]
        sumWidth += x 
        y = im.size[1]
        #The largest height among all the images in the pyramid is set to the height of the frame
        if (y >= maxHeight):
            maxHeight = y
            
    #Instantiate a new image frame
    imageFrame = Image.new("L", (sumWidth, maxHeight),"white")
    offset_x = 0
    #Add all the images in the pyramid to the image frame with offset on x axis so they don't overlap
    for im in pyramidList:
        imageFrame.paste(im,(offset_x,0))
        offset_x = offset_x + im.size[0];
    return imageFrame
    

#Q4
#Image TemplateImage -> Image 
#Takes a image and Template image and produces a image that hihglights the correlation between the image and template for a given threshold level
def FindTemplate(pyramid, template, threshold):
    desiredWidth = 15
    originalHeight = template.size[1]
    originalWidth = template.size[0]
    ratio = originalWidth/desiredWidth
    #Resize the template where width = desiredWidth and height is also reduced to fit the original ratio
    template = template.resize((desiredWidth,originalHeight/ratio),Image.BICUBIC)

    #Find all the coordinates in the image that have correlation values with the template > threshold 
    coordList = []
    originalImage = pyramid[0].convert('RGB') 
    for image in pyramid:
        coord = np.where(ncc.normxcorr2D(image, template) > threshold)
        x_coord = coord[0]
        y_coord = coord[1]
        coords = zip(y_coord,x_coord)
        coordList.append(coords)
    
    for imagePos in range(len(coordList)):
        #Take the power of 0.75 to image position in the pyramidso we adjust according to the size of the picture in our pyramid
        scale = 0.75 ** imagePos
        for pixel in coordList[imagePos]:
            #Adjust the pixel according to the size of the image in the pyramid
            center_x = pixel[0] / scale
            center_y = pixel[1] / scale
            #Add or subtract a certain offset to the center coordinate so that we get four different coordinates to make a box
            offset = 15
            left_x = center_x - offset
            right_x = center_x + offset
            up_y = center_y + offset
            bottom_y = center_y - offset
            
            #Draw the lines to specify the pixels on the original image with correlation values > threshold
            draw = ImageDraw.Draw(originalImage)
            
            #Draw the horizontal lines
            draw.line((left_x,up_y,right_x,up_y),fill="red",width=2)
            draw.line((left_x,bottom_y,right_x,bottom_y),fill="red",width=2)
            #Draw the vertical lines
            draw.line((left_x,up_y,left_x,bottom_y),fill="red",width=2)
            draw.line((right_x,up_y,right_x,bottom_y),fill="red",width=2)
            del draw  
    
    return originalImage



#Q5 Finding equal false positives and false negatives
FindTemplate(makePyramid(judyBatsImage,5),templateImage,0.59)
FindTemplate(makePyramid(familyImage,5),templateImage,0.59)
FindTemplate(makePyramid(treeImage,5),templateImage,0.59)
FindTemplate(makePyramid(studentsImage,5),templateImage,0.59)
FindTemplate(makePyramid(fansImage,5),templateImage,0.59)
FindTemplate(makePyramid(sportsImage,5),templateImage,0.59)
    






        
    
    
    
    