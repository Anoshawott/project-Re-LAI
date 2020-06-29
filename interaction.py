#cd C:\Users\anosh\Documents\GitHub\project-lai

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

import joblib
import tensorflow as tf
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC

# # importing os module  
# import os 
# import pprint 
  
# # Get the list of user's 
# # environment variables 
# env_var = os.environ 

  
# # Print the list of user's 
# # environment variables 
# print("User's Environment variable:") 
# pprint.pprint(dict(env_var), width = 1) 

# Following is for controlling the mouse using python

mouse = Controller()
# TO GIVE CURRENT POSITION

# mouse.position

# TO MOVE TO EXACT GIVEN PIXEL COORDINATE

# mouse.position = (x,y)

# TO MOVE RELATIVE TO THE PROVIDED PARAMETERS OF X AND Y

# mouse.move(x,y)

# MOUSE BUTTON CLICKS (LEFT-PARAM: Type of click, RIGHT-PARAM: number of clicks)

# mouse.click(Button.right, 1)

# SCROLLING

# mouse.scroll(up/down, left/right)










# Following attempts to read and interpret on-screen information

# screenshot() returns an image of some given area
def screenshot(x, y, width, height
               , reduction_factor = 1
               , gray = True
              ):
    
    with mss.mss() as sct:
        # The screen part to capture
        region = {'left': x, 'top': y, 'width': width, 'height': height}

        # Grab the data
        img = sct.grab(region)

        if gray:
            result = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
        else:
            result = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

        img = result[::reduction_factor, ::reduction_factor]
        return img  

# img = screenshot(975, 718, 50, 20)
# cv2.imshow('img', img)
# cv2.waitKey()





#### Digit Recognition 2.0
def get_coins():
    # loop through each number to see if there is a match, a match is appended to a
    # dictionary with its key as the number and the its value as the position of the pt.
    # Then sort the dictionary in increasing order and retrieve the keys to get number
    # With numbers that have the same numbers just have an underscore proceeding the
    # the actual number and then just a placeholder after it to indicate that another
    # number of the same one exists.

    # YOUUU got THISSS!
    screenshot()


    return











# Reading real-time onscreen data

def realtime_capture(wincap):
    loop_time = time()
    while(True):
        
        screenshot = wincap.get_screenshot()
        
        cv2.imshow('Computer Vision', screenshot)

        # debugging the loop rate
        print('FPS {}'.format(1/(time()-loop_time)))
        loop_time = time()

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    return print('Done.')

# wincap = WindowCapture('League of Legends (TM) Client')

# realtime_capture(wincap)








# Digit recognition for reading numbers off the GUI

# mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam',
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])

# checkpoint_path = "models/digit_recognition/digit_training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)
# model.fit(x_train, y_train, epochs=3, callbacks = [cp_callback])


# Already saved a trained model above; accessing checkpoint below
# latest=tf.train.latest_checkpoint(checkpoint_dir)
# model.load_weights(latest)
# loss, acc = model.evaluate(x_test, y_test)
# print("Resotred model, accuracy: {:5.2f}%".format(100*acc))

# Saving entire model
# model.save('models/digit_recognition')

# Having completed model successfully load model here
# new_model = tf.keras.models.load_model('models/digit_recognition')
# loss, acc = new_model.evaluate(x_test, y_test)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# print('digit model passed')


# def digit_detection():
#     # Load the classifier
#     # clf = joblib.load("digits_cls.pkl")

#     # Read the input image 
#     im = cv2.imread("score_test.jpg")

#     # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
#     # im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

#     lower_yellow = np.array([0, 70, 0])
#     upper_yellow = np.array([255, 255, 255])

#     mask = cv2.inRange(im_gray, lower_yellow, upper_yellow)
#     res = cv2.bitwise_and(im, im, mask = mask)

#     cv2.imshow('im', im)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)

#     cv2.waitKey()

#     # Getting HSV range
#     # upper_y = np.uint8([[[74,141,134]]])
#     # hsv_upper_y = cv2.cvtColor(upper_y,cv2.COLOR_BGR2HSV)
#     # print( hsv_upper_y )

#     print('Yesss!')

#     # Inversing the filter
#     ret, im_th = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours in the image
#     ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Get rectangles contains each contour
#     rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # #create an empty image for contours
    # img_contours = np.zeros(im.shape)
    # # draw the contours on the empty image
    # cv2.drawContours(img_contours, ctrs, -1, (0,255,0), 3)
    # cv2.imshow('contours', img_contours)
    # cv2.waitKey()



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
    #         roi = tf.keras.utils.normalize(roi, axis=1)
    #         cv2.imshow("roi", roi)
    #         cv2.waitKey()
    #         # Calculate the HOG features
    #         nbr = new_model.predict([np.array([roi])])
    #         print(nbr)
    #         print(np.argmax(nbr))
    #         cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    #     except:
    #         continue
        
        # print('---------')
        
        

    # cv2.imshow("Resulting Image with Rectangular ROIs", im)
    # cv2.waitKey()

# digit_detection()








# Works in combination with screen get_score()* will interpret the image through OCR


# BELOW IS AN ATTEMPT TO READ FROM TEXT OF GUI
# def get_coins(threshold = 190, var = 10):

#     # got_it = False
    
#     # while not got_it:
        
#         # if is_dead():
#         #     return -1

#     img = screenshot(1122, 868, 50, 20, reduction_factor = 1
#                , gray = False)

#     # cv2.imshow('img', img)
#     # cv2.waitKey()

#     # kernel = np.ones((2,2), np.uint8)
#     # img = cv2.dilate(img, kernel, iterations=1)


#     # for rows in range(0, len(img)):
#     #     for cols in range(0, len(img[0])):
#     #         x = img[rows][cols]
#     #         if x[0] > threshold and x[1] > threshold and x[2] > threshold and max(x) - min(x) < var:
#     #             img[rows][cols] = [0, 0, 0]
#     #         else:
#     #             img[rows][cols] = [255,255,255]

#     # cv2.imshow('img', img)
#     # cv2.waitKey()
#     custom_config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
#     pytesseract.pytesseract.tesseract_cmd = r'C:\Users\anosh\AppData\Local\Tesseract-OCR\tesseract.exe'
#     score = pytesseract.image_to_string(img, config=custom_config)
#     print(score)

#     score = re.sub("[^0-9]", "", score)

#     if score != "":            
#         score = int(score)
#         return score

# NOTE(NOT STABLE IMAGE KEEPS MOVING; NOT FIXED) BELOW IS AN ATTEMPT TO READ FROM THE NUMBER OF GREEN PIXELS IN THE HP BAR
# def get_hp():
#     img = screenshot(859,385,105,5, reduction_factor=1,
#         gray = False)
#     print(img)
#     cv2.imshow('img', img)
#     cv2.waitKey()

# get_hp()

# get_coins()

# thing = False
# while not thing:
#     print(get_coins())
#     time.sleep(2)





# Check if dead
def is_dead():
    return None










# Applying a random position movement to player
#TODO: will need to create a function that returns window_center once figured out how to
# automate resizing and positioning of the application window

# def move_to_radians(radians, radius = 50):
    
#     mouse.move(947 + 50 * math.cos(radians)
#                , 632 + 50 * math.sin(radians))
    
#     return radians


# # Starting game actions

# def buy_start_items():
#     mouse.position = (768, 282)
#     mouse.click(Button.left, 1)
#     time.sleep(1)

#     mouse.position = (526, 247)
#     mouse.click(Button.left, 1)
#     mouse.click(Button.right, 1)
#     time.sleep(1)
#     mouse.position = (580, 247)
#     mouse.click(Button.left, 1)
#     mouse.click(Button.right, 1)
#     time.sleep(1)
#     mouse.position = (626, 247)
#     mouse.click(Button.left, 1)
#     mouse.click(Button.right, 1)
#     time.sleep(1)

#     PressKey(ESC)
#     ReleaseKey(ESC)
#     time.sleep(1)

# def top_game_start():
#     for i in list(range(4))[::-1]:
#         print(i+1)
#         time.sleep(1)
    
#     buy_start_items()

#     PressKey(Y)
#     ReleaseKey(Y)
#     time.sleep(1)

#     mouse.position = (1438, 744)
#     mouse.click(Button.right, 1)
#     time.sleep(30)

# top_game_start()

# Finishing resizing the window and creating automating process to reset the game when each
# episode is complete


# def game_end():
#     mouse.position = (947, 632)
#     mouse.click(Button.left, 1)


# #     outer loop to control the number of data points gathered
#     for i in range(0, 50000):
        
# #         if dead, reset the score and start a new game
#         if is_dead():
#             print('You are dead.  Starting the game over.')
#             start_game()
#             score = 10
#             time.sleep(1)
#         else:
            
# #             randomly change direction
#             direction = move_to_radians(random.uniform(0, math.pi*2), random.uniform(0,300))

print('All works!')



##### OLD EROSION TECHNIQUE

# Erode image to create more distinct numbers
    # kernel = np.ones((4,4), np.uint8)
    # img_erosion = cv2.erode(im_gray, kernel, iterations=1)
    # print(img_erosion)
    # Threshold the image
    # ret, im_th = cv2.threshold(img_erosion, 57, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh', im_th)