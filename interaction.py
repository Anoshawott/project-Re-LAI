#cd C:\Users\anosh\Documents\GitHub\project-lai

import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import mss
import mss.tools
import cv2
import pytesseract
import re

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

img = screenshot(975, 718, 50, 20)
cv2.imshow('img', img)
cv2.waitKey()

# Works in combination with screen get_score()* will interpret the image through OCR

def get_score(threshold = 190, var = 10):

    # got_it = False
    
    # while not got_it:
        
        # if is_dead():
        #     return -1

    img = screenshot(975, 718, 50, 20)
    cv2.imwrite("test.png", img)
    img = cv2.imread('test.png')
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # for rows in range(0, len(img)):
    #     for cols in range(0, len(img[0])):
    #         x = img[rows][cols]
    #         if x[0] > threshold and x[1] > threshold and x[2] > threshold and max(x) - min(x) < var:
    #             img[rows][cols] = [0, 0, 0]
    #         else:
    #             img[rows][cols] = [255,255,255]
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\anosh\AppData\Local\Tesseract-OCR\tesseract.exe'
    score = pytesseract.image_to_string(processed_img)
    print(score)
    
    if score != "":            
        score = int(score)
        return score

print(get_score())
cv2.imshow('img', img)
cv2.waitKey()



print('All works!')