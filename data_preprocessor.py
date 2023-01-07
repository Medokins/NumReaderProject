import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from settings import *

# this need to be rewritten bcs it distors images way to much
def preprocessImage(imageArray: np.ndarray) -> np.ndarray:
    # filing border with white pixels untill it's a square to not distort image while resizing later
    length = max(imageArray.shape[0:2]) + 5
    squared_image = np.empty((length, length), np.uint8)
    squared_image.fill(255)
    ax,ay = (length - imageArray.shape[1]) // 2, (length - imageArray.shape[0])//2
    squared_image[ay:imageArray.shape[0] + ay, ax:ax + imageArray.shape[1]] = imageArray

    img = cv2.resize(squared_image, (28, 28))
    _, blackAndWhiteImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    inverted = np.full(blackAndWhiteImage.shape,255) - blackAndWhiteImage
    inverted = inverted.astype(np.uint8)

    tf_image = np.empty((1, 28, 28), dtype=np.double)
    tf_image[0] = inverted
    # plt.imshow(tf_image[0], cmap = plt.cm.binary)
    # plt.show()
    return tf_image

def getBoundingBoxes(img: np.ndarray, visualize: bool = False, live_view: bool = False) -> np.ndarray:
    """
    :param np.ndarray img: input image
    :return np.ndarray: coordinates of bounding boxes -> [[x_start, y_start, x_end, y_end], [], ...]
    """
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    im = ~im
    binaryIm = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -1)

    # get individual objects, fill in some noise
    num_of_objects, labeledImage, componentStats, _ = cv2.connectedComponentsWithStats(binaryIm)
    remainingComponentLabels = [i for i in range(1, num_of_objects) if componentStats[i][4] >= MIN_AREA_FILTER]
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # final clearing of any artefacts by closing
    maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    closingImage = cv2.morphologyEx(filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, CLOSING_INTERATIONS, cv2.BORDER_REFLECT101)

    # bounding boxes
    contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
    boundRect = []

    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            # this is to filter out small artifacts
            area = cv2.boundingRect(contours_poly[i])[2] * cv2.boundingRect(contours_poly[i])[3] 
            if area > 2000 and area < 10000 and live_view:
                boundRect.append(cv2.boundingRect(contours_poly[i]))
            elif not live_view:
                boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Draw the bounding boxes on the binarized input image:
    if visualize:
        imCopy = img.copy()
        for i in range(len(boundRect)):
            color = (0, 255, 0)
            cv2.rectangle(imCopy, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.imshow('bounding boxes', imCopy)

    coordinates_array = np.empty((len(boundRect), 4), dtype=np.int16)

    for i in range(len(boundRect)):
        # Get the start coordinates and size of bounding box
        x_start, y_start, width, height = boundRect[i]

        coordinates_array[i][0] = x_start
        coordinates_array[i][1] = y_start
        coordinates_array[i][2] = x_start + width 
        coordinates_array[i][3] = y_start + height
    
    return coordinates_array

def readDigits(coordinates_array: np.ndarray, img: np.ndarray):
    model = tf.keras.models.load_model('numReader.model')
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sortedBoundigBoxes = coordinates_array[coordinates_array[:,0].argsort()]
    nums = []
    for index in range(len(sortedBoundigBoxes)):
        single_digit = im[sortedBoundigBoxes[index][1]:sortedBoundigBoxes[index][3],
                          sortedBoundigBoxes[index][0]:sortedBoundigBoxes[index][2]]

        # plt.imshow(single_digit, cmap = plt.cm.binary)
        # plt.show()
        boundingBoxConverted = preprocessImage(single_digit)
        nums.append(np.argmax(model.predict([boundingBoxConverted])))
    
    for number in nums:
        print(number, end=' ')