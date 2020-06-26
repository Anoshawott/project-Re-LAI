#cd /mnt/c/Users/anosh/Documents/GitHub/project-lai/

import numpy as np
import pyautogui
import quartz

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

# Following class is for controlling the mouse using python

class Mouse():
    down = [quartz.kCGEventLeftMouseDown, quartz.kCGEventRightMouseDown, quartz.kCGEventOtherMouseDown]
    up = [quartz.kCGEventLeftMouseUp, quartz.kCGEventRightMouseUp, quartz.kCGEventOtherMouseUp]
    [LEFT, RIGHT, OTHER] = [0, 1, 2]

    def position(self):
        point = quartz.CGEventGetLocation( quartz.CGEventCreate(None) )
        return point.x, point.y

    def location(self):
        loc = NSEvent.mouseLocation()
        return loc.x, quartz.CGDisplayPixelsHigh(0) - loc.y

    def move(self, x, y):
        moveEvent = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventMouseMoved, (x, y), 0)
        quartz.CGEventPost(quartz.kCGHIDEventTap, moveEvent)

    def press(self, x, y, button=1):
        event = quartz.CGEventCreateMouseEvent(None, Mouse.down[button], (x, y), button - 1)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)

    def release(self, x, y, button=1):
        event = quartz.CGEventCreateMouseEvent(None, Mouse.up[button], (x, y), button - 1)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)

    def click(self, button=LEFT):
        x, y = self.position()
        self.press(x, y, button)
        self.release(x, y, button)

    def click_pos(self, x, y, button=LEFT):
        self.move(x, y)
        self.click(button)

    def to_relative(self, x, y):
        curr_pos = quartz.CGEventGetLocation( quartz.CGEventCreate(None) )
        x += current_position.x;
        y += current_position.y;
        return [x, y]

    def move_rel(self, x, y):
        [x, y] = to_relative(x, y)
        moveEvent = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventMouseMoved, quartz.CGPointMake(x, y), 0)
        quartz.CGEventPost(quartz.kCGHIDEventTap, moveEvent)

print('All works!')