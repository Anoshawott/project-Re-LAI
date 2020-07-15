import pickle
import time
import math
from pynput.mouse import Button, Controller
from RealRecognition import Detection
from window_cap import WindowCapture
from DirectKeys import PressKey, ReleaseKey, Y, Q, W, E, R, D, F, B, ESC, CONTROL, num_1, num_2, num_3, num_4, num_5, num_6 

class PlayerAI:
    def __init__(self):
        self.data = Detection().get_data()
        # self.playerai_stats = self.data['output_data']
        self.map_stats = self.data['map_data']
        # self.pot_enemey = self.data['enemy_presence']
        self.x_map, self.y_map = self.map_stats['player_pos']
        self.win_prop = WindowCapture('League of Legends (TM) Client').get_screen_position()
        self.choices = pickle.load(open('choices.pickle', 'rb'))
        # self.x, self.y = '' --> maybe something to think about later...
    
    def new_data(self):
        return Detection().get_data()

    def action(self, choice):
        # each choice should be associated with a chosen position of the 
        # mouse position choices will range from 5 radii each and have 36 different
        # movement positions within the radius --> therefore 5*36 = 180 choices...

        # MAKE A FUNCTION THAT TAKES A CHOICE NUMBER WHICH IS ASSIGNED TO A VALUE IN 
        # A LIST THAT GIVES THE RADII AND RADIAN POSITION ON THE GIVEN CIRCLE...(refer to choice_create.py)

        # Reading choices dictionary made from choices_create.py
        if choice <= 1259:
            self.move(r=self.choices[choice][0], deg=self.choices[choice][1], 
                        ability=self.choices[choice][2])
        else:
            self.sub_actions(choice=self.choices[choice])
        
    def sub_actions(self, choice=None):
        if len(choice) > 1:
            PressKey(choice[0])
            PressKey(choice[1])
            ReleaseKey(choice[0])
            ReleaseKey(choice[1])
        else:
            if choice[0] == B:
                PressKey(choice[0])
                time.sleep(3)
                ReleaseKey(choice[0])
                time.sleep(10)
            else:
                PressKey(choice[0])
                ReleaseKey(choice[0])
        return

    def move(self, r=None, deg=None, ability=None):
        win_center_x = self.win_prop[0] + self.win_prop[2]/2
        win_center_y = self.win_prop[1] + self.win_prop[3]/2

        mouse = Controller()

        mouse_x = win_center_x + r*math.cos(deg*(math.pi/180))
        mouse_y = win_center_y + r*math.sin(deg*(math.pi/180))

        mouse.position = (mouse_x, mouse_y)
        mouse.click(Button.right, 1)
        
        if ability == '':
            return
        else:
            PressKey(ability)
            ReleaseKey(ability)
            return
