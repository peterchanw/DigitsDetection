from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam

def digitsModel(imgDimension, classNo):
    filterNo = 60
    filter1Size = (5,5)
    filter2Size = (3,3)
    poolSize = (2,2)
    nodeNum = 500

    model = Sequential()
    model.add(Conv2D(filterNo, filter1Size, input_shape=(imgDimension[0],imgDimension[1],1), activation= 'relu'))
    model.add(Conv2D(filterNo, filter1Size, activation='relu'))
    model.add(MaxPooling2D(pool_size=poolSize))
    model.add(Conv2D(filterNo // 2, filter2Size, activation='relu'))
    model.add(Conv2D(filterNo // 2, filter2Size, activation='relu'))
    model.add(MaxPooling2D(pool_size=poolSize))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(nodeNum, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classNo, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])
    return model




