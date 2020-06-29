import numpy as np
from DirectKeys import PressKey, ReleaseKey, Y, ESC
import pyautogui
from PIL import Image
from pynput.mouse import Button, Controller
import mss
import mss.tools
import cv2
import pytesseract
import re
import math
import time
import random
from window_cap import WindowCapture
import os
    
    
# Read the input image 
im = cv2.imread("score_test.jpg")

edges = cv2.Canny(im,100,50)
cv2.imshow('edges', edges)
cv2.waitKey()
cv2.imwrite('edges.jpg', edges)

# Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

lower_yellow = np.array([0, 70, 0])
upper_yellow = np.array([255, 255, 255])

mask = cv2.inRange(im_gray, lower_yellow, upper_yellow)
res = cv2.bitwise_and(im, im, mask = mask)

# cv2.imshow('im', im)
# cv2.imshow('mask', mask)
# cv2.imshow('res', res)
# cv2.waitKey()


# Getting HSV range
# upper_y = np.uint8([[[74,141,134]]])
# hsv_upper_y = cv2.cvtColor(upper_y,cv2.COLOR_BGR2HSV)
# print( hsv_upper_y )

print('Yesss!')

ret, im_th = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow('mask_edit', im_th)
# cv2.waitKey()

kernel = np.ones((2,2), np.float32)/4
# smoothed = cv2.filter2D(im_th,-1,kernel)

# blur = cv2.GaussianBlur(im_th, (5,5), 0)
# median = cv2.medianBlur(im_th, 3)

# k = np.ones((1,1), np.uint8)
# median_dilation = cv2.erode(median, k, iterations=1)

# cv2.imwrite('score_edit.jpg', im_th)

opening = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)

# cv2.imshow('blur', blur)
# cv2.imshow('median', median)
# cv2.imshow('median_dilation', median_dilation)
cv2.imshow('opening', opening)
# cv2.imshow('im_th', im_th)

cv2.waitKey()

# new_image = cv2.imread("score_edit.jpg")
# ret, new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY)
# cv2.imshow('new_iamge',im_th)
# cv2.waitKey()

# # Find contours in the image
# ctrs, hier = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Get rectangles contains each contour
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# # For each rectangular region, calculate HOG features and predict
# # the digit using Linear SVM.
# for rect in rects:
#     # Draw the rectangles
#     cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
#     # Make the rectangular region around the digit
#     leng = int(rect[3] * 1.6)
#     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
#     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
#     roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
#     # Resize the image
#     try:
#         roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#         roi = cv2.dilate(roi, (3, 3))
#         # roi = tf.keras.utils.normalize(roi, axis=1)
#         cv2.imshow("roi", roi)
#         cv2.waitKey()
#         # Calculate the HOG features
#         # nbr = new_model.predict([np.array([roi])])
#         # print(nbr)
#         # print(np.argmax(nbr))
#         # cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
#     except:
#         continue

# cv2.imshow("Resulting Image with Rectangular ROIs", im)
# cv2.waitKey()