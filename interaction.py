#cd C:\Users\anosh\Documents\GitHub\project-lai

import numpy as np
from DirectKeys import PressKey, ReleaseKey, Y, ESC
import pyautogui
from PIL import ImageGrab
from pynput.mouse import Button, Controller
import mss
import mss.tools
import cv2
import pytesseract
import re
import math
from time import time
import random
from window_cap import WindowCapture

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

wincap = WindowCapture('League of Legends (TM) Client')

realtime_capture(wincap)











# Works in combination with screen get_score()* will interpret the image through OCR

def get_score(threshold = 190, var = 10):

    # got_it = False
    
    # while not got_it:
        
        # if is_dead():
        #     return -1

    img = screenshot(883, 850, 30, 20)
    cv2.imshow('img', img)
    cv2.waitKey()

    # for rows in range(0, len(img)):
    #     for cols in range(0, len(img[0])):
    #         x = img[rows][cols]
    #         if x[0] > threshold and x[1] > threshold and x[2] > threshold and max(x) - min(x) < var:
    #             img[rows][cols] = [0, 0, 0]
    #         else:
    #             img[rows][cols] = [255,255,255]
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\anosh\AppData\Local\Tesseract-OCR\tesseract.exe'
    score = pytesseract.image_to_string(img)

    score = re.sub("[^0-9]", "", score)

    if score != "":            
        score = int(score)
        return score

get_score()




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