#cd /mnt/c/Users/anosh/Documents/GitHub/project-lai/

import numpy as np
import pyautogui
import quartz
from pynput.mouse import Button, Controller
import mss
import mss.tools
import cv2

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

cv2.imshow('img', img)
cv2.waitkey()





print('All works!')