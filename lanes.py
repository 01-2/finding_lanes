import cv2
import numpy as np
# import matplotlib.pyplot as plt

def canny(image):
    '''
    #2 : Grayscale Conversion
        Edge Detection : identifying sharp changes in intensity in adjacent pixels
        Gradient : Measure of change in brightness over adjacent pixels

        edge : rapid changes in brightness

        step 1 : gray scale
            3 channels to 1 channel(one intensity)
            = fast processing
    '''
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    '''
    #3 : Gaussian Blur
        step 2. Reduce Noise
    '''
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    '''
    #4 : Canny
        derivate(f(x,y)) -> measure adjacent changes in intensity in all directions, x and y
        cv2.Canny(image, low_threshold, high_threshold)
    '''
    canny = cv2.Canny(blur, 50, 150)

    return canny


'''
#5 Region of Interest
    using pyplot, setting region of interest
'''
def region_of_interest(image):
    height = image.shape[0]
    polygons= np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)

    '''
    #6 Bitwise_and
        masking region of interest at lane image
    '''
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

'''
#1 : Installation
    imread, imshow
'''

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
#plt.imshow(canny)
#plt.show()

'''
#7 Hough Transform
    finding possible line by hough space 
    can determine ax + b
'''

cv2.imshow("result", cropped_image)
cv2.waitKey(0)
