#cd C:\Users\anosh\Documents\GitHub\project-lai

import numpy as np
import cv2
from window_cap import WindowCapture
import time
import math

##### SUCCESS!!!!!!!
# NOTE: 0.8 = get_coins, GET_HP WILL NEED ITS OWN SET OF NUMBERS TO COMPARE, ALSO NUMBERS FOR
# METRICS MAY BE DIFFERENT AND MAY NEED EXTRA SET 

class Detection:

    # screenshot() returns an image of some given area
    def screenshot(self):
        wincap = WindowCapture('League of Legends (TM) Client')
        img = wincap.get_screenshot()
        return img  

    # returns digit and position
    def digit_detect(self, img, copy=None, region=None, x=0, y=0, width=1270, height=711):
        tmp_numbers = {}
        # Read and crop the input image 
        crop_img = img[y:y+height, x:x+width]
        
        # if x == 146 and y==20 and width==30 and height==15:
        #     cv2.imwrite('hp_test.jpg', crop_img)

        # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
        im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('images/numbers/'+ region +'/' + str(copy) + '.jpg', 0)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
        if region == 'hp':
            threshold = 0.9
        # elif region == 'hp-tur':
        #     threshold = 0.77
        else:
            threshold = 0.8
        
        loc = np.where(res >= threshold)

        count = 0
        for pt in zip(*loc[::-1]):
            if count == 1:
                tmp_numbers[str(copy)+'_!'] = pt[0]
                count += 1
            elif count == 2:
                tmp_numbers[str(copy)+'_@'] = pt[0]
                count += 1
            elif count == 3:
                tmp_numbers[str(copy)+'_#'] = pt[0]
            else:
                tmp_numbers[str(copy)] = pt[0]
                count += 1
        return tmp_numbers


    # returns player position from mini-map
    def player_detect(self, img, copy=None, region=None, x=0, y=0, width=1270, height=711, last = None):
        tmp_numbers = {}
        # Read and crop the input image 
        crop_img = img[y:y+height, x:x+width]
        # cv2.imwrite('mini_map.jpg', crop_img)

        # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
        im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('images/players/player_control.jpg', 0)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
        threshold = 0.45
        
        loc = np.where(res >= threshold)
        x_pos = last[0]
        y_pos = last[1]

        for pt in zip(*loc[::-1]):
            # cv2.rectangle(crop_img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)
            x_pos = pt[0]+w/2
            y_pos = pt[1]+h/2
        return x_pos, y_pos
    
    # returns enemy position from surroundings
    def enemy_detect(self, img, copy=None, region=None, x=0, y=0, width=1270, height=711, last = None):
        tmp_numbers = {}
        # Read and crop the input image 
        # crop_img = img[y:y+height, x:x+width]
        # cv2.imwrite('mini_map.jpg', crop_img)

        # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('images/enemy/6.jpg', 0)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
        threshold = 0.8
        
        loc = np.where(res >= threshold)
        # x_pos = last[0]
        # y_pos = last[1]

        count = 0
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 1)
            count+=1
        
        if count > 1:
            return 1
        else:
            return 0
            
        

    # returns turret state and distance from player from mini-map
    # def turret_detect(self, img, copy=None, region=None, x=0, y=0, width=1270, height=711):
    #     tmp_numbers = {}
    #     # Read and crop the input image 
    #     crop_img = img[y:y+height, x:x+width]

    #     # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
    #     im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #     template = cv2.imread('images/players/player_control.jpg', 0)

    #     w, h = template.shape[::-1]
    #     res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
    #     threshold = 0.45
        
    #     loc = np.where(res >= threshold)
    #     x_pos = None
    #     y_pos = None

    #     for pt in zip(*loc[::-1]):
    #         # cv2.rectangle(crop_img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 2)
    #         x_pos = pt[0]/2
    #         y_pos = pt[1]/2
    #     return x_pos, y_pos, crop_img


    # put output_data variable for last_digits when initiating function; turret':[146,20,30,15],
    def get_data(self, last_digits = None, last_pos = None, time = 0):
        img = self.screenshot()
        cv2.imwrite('test.jpg', img)
        # cv2.imwrite('turret_display.jpg', img)
        input_data = {'cs':[1178,0,25,16], 
                'kda':[1103,0,43,13], 'level':[402,684,25,25], 
                'hp':[550,680,36,16], 'coins': [797,691,51,17],
                 'map':[1087,527,181,181]}
        output_data = {}
        map_data = {}
        time = math.floor(time)

        # Only reading coin data every 50 seconds
        if time%50 != 0:
            del input_data['coins']
        
        for area in input_data:
            # print(area,'----------')
            x_coor = input_data[area][0]
            y_coor = input_data[area][1]
            w = input_data[area][2]
            h = input_data[area][3]
            numbers = {}
            if area == 'cs' or area=='kda' or area=='level' or area=='hp' or area=='coins' or area=='turret':
                for number in range(10):
                    # print(number)
                    returned_numbers = self.digit_detect(img = img, copy = number, region = area,
                                                x = x_coor, y = y_coor, width = w, height = h)
                    z = numbers.copy()
                    z.update(returned_numbers)
                    numbers = z
                num = {k: v for k, v in sorted(numbers.items(), key=lambda item: item[1])}
                num_str = ''
                for key in num:
                    num_str = num_str + key
                remove = ['_', '!', '@', "#"]
                for i in remove:
                    num_str = num_str.replace(i, '')
                if area == 'kda' and num_str != '':
                    kda = []
                    for i in num_str:
                        if i == '':
                            return last_digits
                        else:
                            kda.append(i)
                    output_data[area]=kda
                    continue
                if num_str == '':
                    output_data[area]=last_digits[area]
                else:
                    output_data[area]=num_str
            else:
                # fix last thiing later just a minor problem easy when formulating final player positions
                # in player_interaction.py
                
                if last_pos is None:
                    last = [20,170]
                else:
                    last = last_pos

                x_pos, y_pos = self.player_detect(img = img, copy = number, region = area,
                                                x = x_coor, y = y_coor, width = w, height = h, last=last)
                map_data['player_pos']=[x_pos, y_pos]

                # closest distance coordinate to turret_outer --> 6 units
                # Try get this distance and apply this same distance condition for the other turrets 

                #### Just get the other turret coordinates from their centre in photoshop --> then figure out
                #### how to determine which turret hp information will be updated from an EXISTING DICT

                turrets = {'tur_outer':[112,86],'tur_inner':[122,66],'tur_inhib':[138,53],
                            'inhib':[143,46],'tur_nex_1':[155,30],'tur_nex_2':[160,36],'nexus':[162,28]}

                def dist(cur_pos = None, tur_pos = None):
                    return math.sqrt((tur_pos[1]-cur_pos[1])**2 + (tur_pos[0]-cur_pos[0])**2)
                
                map_data['tur_dist']={}
                map_data['tur_hp']={}

                x_coor = 146
                y_coor = 20
                w = 30
                h = 15
                
                for tur in turrets:
                    rel_dist = dist(map_data['player_pos'],turrets[tur])
                    map_data['tur_dist'][tur]=math.floor(rel_dist)
                    if math.floor(rel_dist)<=7:
                        for number in range(10):
                            # print(number)
                            returned_numbers = self.digit_detect(img = img, copy = number, region = 'hp-tur',
                                                        x = x_coor, y = y_coor, width = w, height = h)
                            z = numbers.copy()
                            z.update(returned_numbers)
                            numbers = z
                        num = {k: v for k, v in sorted(numbers.items(), key=lambda item: item[1])}
                        num_str = ''
                        for key in num:
                            num_str = num_str + key
                        remove = ['_', '!', '@', "#"]
                        for i in remove:
                            num_str = num_str.replace(i, '')
                        if num_str == '':
                            None
                        else:
                            map_data['tur_hp'][tur]=num_str

                # have a HIGHER REWARD for getting closer and attacking higher up turrets 
                ##### need to do MANA check and reward system for regaining this but not too much that it 
                ##### does it constantly without pushing forward
                # Only reading turret position a certain distance away --> only store turret HP if a 
                # value for turret HP is established

            # Enemy detect sequence; read only 1270x560
            ## HAVE FACIAL RECOGNITION OF PLAYERS ON START SCREEN; SAVE EACH PLAYER FACE FROM LOADING
            ## SCREEN. IF THERE ARE MORE THAN ONE LEVEL BARS IN THE GIVEN SCREEN AI WILL QUICKLY CLICK 
            ## ON THE OTHER PLAYER THAT HAS ENTERED THE SCREEN IDENTIFY FROM THE TOP LEFT SPACE THE FACE
            ## USING TEMPLATE MATCHING WILL UNDERSTAND IF THE PLAYER IS AN ENEMY OR ALLY!!!!!

            ### OR try to analyse dataset from online --> look into...

            ### FINAL DECISION: Try read enemy hud in top left use hsv range colour check
            ### to see if red if in the red range then return as enemy within range...
            # (WILL COME BACK TO LATER SINCE TAKING WAY TOO MUCH TIME RIGHT NOW!!)

            

            ### SIDE NOTE: might need to consider distance to other turrets; strategy may change throughout 
            # the game where champion should go based on other players...

            # FOR NOW --> 
            # Just to know that there are potential enemies around ; take caution...
            enemy_presence = self.enemy_detect(img=img)

        return {'output_data':output_data, 'map_data':map_data,
                'enemy_presence': enemy_presence}


# Following attempts to read and interpret on-screen information 
#### (Digit Recognition 3.0)

# for i in list(range(3))[::-1]:
#     print(i+1)
#     time.sleep(1)

# start = time.time()
# count = 0
# while True:
#     if count == 0:
#         print('Time elapsed:', time.time()-start)
#         print(DigitDetect().get_data())
#         print('-------------')
#         last_data = DigitDetect().get_data()

#         count += 1
#     else:
#         print('Time elapsed:', time.time()-start)
#         start_1 = time.time()
#         print(DigitDetect().get_data(last = last_data))
#         print(time.time()-start_1)
#         print('-------------')
#         last_data = DigitDetect().get_data()


#     time.sleep(5)



# Digit recognition 2.5 --> with new windowed coordinates but with inefficient data collection

# for i in list(range(3))[::-1]:
#     print(i+1)
#     time.sleep(1)

# start = time.time()
# count = 0
# while True:
#     if count == 0:
#         print('Time elapsed:', time.time()-start)
#         print('Coins: ', DigitDetect().get_data(797,691,51,17, where = 'coins'))
#         print('CP: ', DigitDetect().get_data(1178,0,25,16, where = 'cp'))
#         print('KDA: ', DigitDetect().get_data(1103,0,43,13, where = 'kda'))
#         print('-------------')
#         last_coins = DigitDetect().get_data(797,691,51,17, where = 'coins')
#         last_cp = DigitDetect().get_data(1178,0,25,16, where = 'cp')
#         last_kda = DigitDetect().get_data(1103,0,43,13, where = 'kda')

#         count += 1
#     else:
#         print('Time elapsed:', time.time()-start)
#         start_1 = time.time()
#         print('Coins: ', DigitDetect().get_data(797,691,51,17, where = 'coins', last = last_coins))
#         print(time.time()-start_1)
#         print('CP: ', DigitDetect().get_data(1178,0,25,16, where = 'cp', last = last_cp))
#         print('KDA: ', DigitDetect().get_data(1103,0,43,13, where = 'kda', last = last_kda))
#         print('-------------')
#         last_coins = DigitDetect().get_data(797,691,51,17, where = 'coins')
#         last_cp = DigitDetect().get_data(1178,0,25,16, where = 'cp')
#         last_kda = DigitDetect().get_data(1103,0,43,13, where = 'kda')


#     time.sleep(5)

#below uses coordinates when the whole screen was being read...

# start = time.time()
# count = 0
# while True:
#     if count == 0:
#         print('Time elapsed:', time.time()-start)
#         coins = DigitDetect().get_data(1117,868,51,17, where = 'coins')
#         cp = DigitDetect().get_data(1504,173,25,16, where = 'cp')
#         kda = DigitDetect().get_data(1427,172,43,13, where = 'kda')
#         print('-------------')
#         last_coins = coins
#         last_cp = cp
#         last_kda = kda

#         count += 1
#     else:
#         print('Time elapsed:', time.time()-start)
#         coins = DigitDetect().get_data(1117,868,51,17, where = 'coins', last = last_coins)
#         cp = DigitDetect().get_data(1504,173,25,16, where = 'cp', last = last_cp)
#         kda = DigitDetect().get_data(1427,172,43,13, where = 'kda', last = last_kda)
#         print('-------------')
#         last_coins = coins
#         last_cp = cp
#         last_kda = kda
        
#     time.sleep(5)


# MAKE SCREENSHOT FASTER USE DIRECT WINDOWS METHOD!!!! get each screengrab down to 0.01 then we can total 
# time with each detection with 0.05 = 20fps
