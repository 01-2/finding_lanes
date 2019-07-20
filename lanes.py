import cv2

''' imread, imshow '''
image = cv2.imread('test_image.jpg')

cv2.imshow('result', image)
cv2.waitKey(0)