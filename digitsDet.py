import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from keras.models import load_model
import utilsImg

# ****************************************************************************
thresholds = 0.60
# ****************************************************************************

cap = VideoStream(src=0).start()

model = load_model('digits_model.h5')

while True:
    # read the frame from webcam
    frame = cap.read()
    frame = imutils.resize(frame, width=320)
    image = frame.copy()
    # prepare the image for the digit prediction CNN model
    image = cv2.resize(image,(32,32))
    image = np.asarray(image)
    image = utilsImg.preProcessing(image)
    image = image.reshape(1,32,32,1)

    # Prediction
    classIndex = int(model.predict_classes(image))      # predict the Class Index
    # print(classIndex)
    predictions = model.predict(image)                  # run model predictions
    probVal = np.amax(predictions)                      # find maximum probability (i.e. likely) of predicted digit
    if probVal > thresholds:
        print('Predicted digit: {} '.format(classIndex), end= ' ')
        print('Probability: {0:.2f}'.format(probVal))
        print('')
        cv2.putText(frame, 'Predict Digit: '+str(classIndex), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(frame, 'Probability: ' + str(round(probVal*100, 2)) + '%', (10,50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)
    else:
        print('Unable to predict any digit!')

    cv2.imshow('Original image', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.stop()