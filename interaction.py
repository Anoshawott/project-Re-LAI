#cd C:\Users\anosh\Documents\GitHub

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

img = screenshot(664, 1022, 15, 15)

cv2.imshow('img', img)
cv2.waitKey()


# Works in combination with screen get_score()* will interpret the image through OCR

def get_score(threshold = 190, var = 10):

    # got_it = False
    
    # while not got_it:
        
        # if is_dead():
        #     return -1

    img = screenshot(x= 664, y= 1022, width = 15, height = 15, gray = False)

    for rows in range(0, len(img)):
        for cols in range(0, len(img[0])):
            x = img[rows][cols]
            if x[0] > threshold and x[1] > threshold and x[2] > threshold and max(x) - min(x) < var:
                img[rows][cols] = [0, 0, 0]
            else:
                img[rows][cols] = [255,255,255]
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'
    score = pytesseract.image_to_string(img, config='outputbase digits')
    
    score = re.sub("[^0-9]", "", score)
    
    if score != "":            
        score = int(score)
        return score

print(get_score())



print('All works!')