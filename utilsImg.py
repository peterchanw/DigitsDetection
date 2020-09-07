import cv2
import imutils
import numpy as np
import os

### prepare import of digit images dataset and store in numpy lists images and corresponding classID
def prepareDataset(path, imgDimensions):
    images = []
    classID = []
    ### create a class list from the directory structure
    className = os.listdir(path)
    print('Classes detected', className)    # Class names list
    classNo = len(className)                # Number of classes

    ### construct the digits dataset
    # create a list to store all the images read from the directory structure with corresponding classID
    print('Import images ...')
    for x in range(0, classNo):
        picList = os.listdir(path + '/' + str(x))
        # print(picList)
        for y in picList:
            curImg = cv2.imread(path + '/' + str(x) + '/' + y)  # read the image
            curImg = imutils.resize(curImg, width=imgDimensions[0])
            images.append(curImg)  # append the image to a list
            classID.append(x)
        print(x, end=' ')  # display result in horizontal way
    print('')  # reset print function
    # print(classID)
    # print(len(classID))
    # print(len(images))
    print('End of import images.')

    ### Create Training dataset, Validation dataset and Test dataset
    # converted into numpy array from list for image processing
    images = np.array(images)
    classID = np.array(classID)
    print('Whole dataset: ', images.shape)
    print('')
    # print(classID.shape)
    return images, classID, className

# print out the number of samples in each class for Training dataset, Validation dataset and Test dataset
def printDataset(X_dataset, y_dataset, classNo):
    Samples = []
    for i in range(0, classNo):
        eachClassTotal = len(np.where(y_dataset == i)[0])
        # print('Training dataset {}: {}'.format(i, eachClassTotal))
        Samples.append(eachClassTotal)
    print(Samples)

# run a function against a list element using map(). In this case, pre-Processing each image in image list and
# store back to image list
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) / 255     # Equalize the brightness evenly for the images and then normalize (0, 1)
    return img

