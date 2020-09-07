# import cv2
# import imutils
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import utilsImg
import model

# ****************************************************************************
path = "data"
picList = []
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 4
epochsVal = 20
stepsPerEpochsVal = 1500
# ****************************************************************************

# STEP 1: prepare import of digit images dataset and store in numpy lists images and corresponding classID
images, classID, className = utilsImg.prepareDataset(path, imageDimensions)
classNo = len(className)                # Number of classes

# STEP 2: splitting the data (Training dataset, Validation dataset and Test dataset)
X_train, X_test, y_train, y_test = train_test_split(images, classID, test_size=testRatio)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size= valRatio)

# print out the number of samples in each class for Training dataset, Validation dataset and Test dataset
# Training dataset
print('Training dataset: ', X_train.shape)
utilsImg.printDataset(X_train, y_train, classNo)
# Validation dataset
print('Validation dataset: ', X_valid.shape)
utilsImg.printDataset(X_valid, y_valid, classNo)
# Test dataset
print('Test dataset: ', X_test.shape)
utilsImg.printDataset(X_test, y_test, classNo)

## display the number of images in each classes in a bar chart
# plt.figure(figsize=(10,5))
# plt.bar(range(0, classNo), trainSamples)
# plt.title('Number of images in each Class - Training dataset')
# plt.xlabel('Class ID')
# plt.ylabel('Number of images')
# plt.show()

## display a particular pre-processed image
# img = utils.preProcessing(X_train[30])
# img = imutils.resize(img, width=300)
# cv2.imshow('Pre-processing image: ', img)
# cv2.waitKey(0)

# STEP 3: run a function against a list element using map(). In this case, pre-Processing each image in X_train list and
# store back to X_train
X_train = np.array(list(map(utilsImg.preProcessing, X_train)))
# print(X_train[30].shape)
# img = X_train[30]
# img = imutils.resize(img, width=300)
# cv2.imshow('Pre-processing image: ', img)
# cv2.waitKey(0)

# run a function against a list element using map(). In this case, pre-Processing each image in validation dataset and
# test dataset
X_valid = np.array(list(map(utilsImg.preProcessing, X_valid)))
X_test = np.array(list(map(utilsImg.preProcessing, X_test)))

# STEP 4: Add depth (i.e. One) for the images for Convolution Neural Network (CNN)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
# print(X_train.shape)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# STEP 5: data augmentation for CNN to generate more samples from existing datasets
# samples shift 10% in width for existing images
# samples shift 10% in height for existing images
# samples zoom in and out 20%
# samples shear in and out 10%
# samples rotate by 30 degree

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2,
                             shear_range=0.2, rotation_range=30)
dataGen.fit(X_train)        # calculate data generator statistics helper and input parameters for the data generator

# STEP 6: one-hot encode integer data
# If the integer data represents all the possible values of the classes,
# then to_categorical() method can be used directly;
# otherwise, the number of classes can be passed to the method as the num_classes parameter.
# y_train = to_categorical(y_train, classNo)
# y_valid = to_categorical(y_valid, classNo)
# y_test = to_categorical(y_test, classNo)

# convert the labels from integers to vectors
y_train = LabelBinarizer().fit_transform(y_train)
y_valid = LabelBinarizer().fit_transform(y_valid)
y_test = LabelBinarizer().fit_transform(y_test)

# STEP 7: construct and compile the CNN model to analysis the datasets
model = model.digitsModel(imageDimensions, classNo)
print(model.summary())

# STEP 8: run the model to predict digits
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                    batch_size=batchSizeVal,
                    steps_per_epoch=stepsPerEpochsVal,
                    epochs=epochsVal,
                    validation_data=(X_valid, y_valid),
                    shuffle=True)

# history = model.fit_generator(dataGen.flow(X_train, y_train,
#                                  batch_size=batchSizeVal),
#                                  steps_per_epoch=stepsPerEpochsVal,
#                                  epochs=epochsVal,
#                                  validation_data=(X_valid, y_valid),
#                                  shuffle=1)

# STEP 9: plot the results to check the accuracy and generalisation of the digits recognition model
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

print("[INFO] evaluating network...")
score = model.evaluate(X_test, y_test, verbose=0)
# print('Test score: {0:.2f}'.format(score[0]))
print('Accuracy: {0:.2f}%'.format(score[1]*100))
print('')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=className))

# STEP 10: Save the model
# save the network to disk
print("[INFO] serializing CNN network...")
model.save("digits_model.h5")
print("[INFO] CNN model saved.")

