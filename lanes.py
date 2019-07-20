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
    step 2. Reduce Noise
'''
blur = cv2.GaussianBlur(gray, (5, 5), 0)

'''
#4 : Finding Lane Lines(Canny)
    derivate(f(x,y)) -> measure adjacent changes in intensity in all directions, x and y
    cv2.Canny(image, low_threshold, high_threshold)
'''
canny = cv2.Canny(blur, 50, 150)


cv2.imshow('result', canny)
cv2.waitKey(0)

