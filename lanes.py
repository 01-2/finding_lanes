import cv2
import numpy as np

'''
#1 : Installation
imread, imshow
'''

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

'''
#2 : Findding Lane Lines(Grayscale Conversion)
Edge Detection : identifying sharp changes in intensity in adjacent pixels
Gradient : Measure of change in brightness over adjacent pixels

edge : rapid changes in brightness

step 1 : gray scale
    3 channels to 1 channel(one intensity)
    = fast processing
'''
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

'''
#3 : Finding Lane Lines(Gaussian Blur)

'''
cv2.imshow('result', gray)
cv2.waitKey(0)

