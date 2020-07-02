import numpy as np
import cv2
from window_cap import WindowCapture
import time

##### SUCCESS!!!!!!!
# NOTE: 0.8 = get_coins, GET_HP WILL NEED ITS OWN SET OF NUMBERS TO COMPARE, ALSO NUMBERS FOR
# METRICS MAY BE DIFFERENT AND MAY NEED EXTRA SET 

class DigitDetect:



    # screenshot() returns an image of some given area
    def screenshot(self):
        wincap = WindowCapture('League of Legends (TM) Client')
        img = wincap.get_screenshot()
        return img  

    # returns digit and position
    def detect(self, img, copy, region, x, y, width, height):
        tmp_numbers = {}
        # Read and crop the input image 
        crop_img = img[y:y+height, x:x+width]
        if region == 'hp':
            cv2.imwrite('hp_test.jpg', crop_img)

        # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
        im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('images/numbers/'+ region +'/' + str(copy) + '.jpg', 0)
        w, h = template.shape[::-1]
        
        res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
        if region == 'hp':
            threshold = 0.9
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

    # 'mini_map_positions':[1087,527,181,181] 'coins': [797,691,51,17],
    def get_data(self, last = ''):
        img = self.screenshot()
        input_data = {'cs':[1178,0,25,16], 
                'kda':[1103,0,43,13], 'level':[402,684,25,25], 
                'hp':[550,680,36,16]}
        output_data = {}
        for area in input_data:
            # print(area,'----------')
            x_coor = input_data[area][0]
            y_coor = input_data[area][1]
            w = input_data[area][2]
            h = input_data[area][3]
            numbers = {}
            for number in range(10):
                # print(number)
                returned_numbers = self.detect(img = img, copy = number, region = area,
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
                        return last
                    else:
                        kda.append(i)
                output_data[area]=kda
                continue
            if num_str == '':
                output_data[area]=last
            else:
                output_data[area]=num_str
        
        return output_data


# Following attempts to read and interpret on-screen information 
#### (Digit Recognition 3.0)

for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)

start = time.time()
count = 0
while True:
    if count == 0:
        print('Time elapsed:', time.time()-start)
        print(DigitDetect().get_data())
        print('-------------')
        last_data = DigitDetect().get_data()

        count += 1
    else:
        print('Time elapsed:', time.time()-start)
        start_1 = time.time()
        print(DigitDetect().get_data(last = last_data))
        print(time.time()-start_1)
        print('-------------')
        last_data = DigitDetect().get_data()


    time.sleep(5)



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
