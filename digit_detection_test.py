import numpy as np
import cv2
import mss
import mss.tools
import time

##### SUCCESS!!!!!!!
# NOTE: 0.8 = get_coins, GET_HP WILL NEED ITS OWN SET OF NUMBERS TO COMPARE, ALSO NUMBERS FOR
# METRICS MAY BE DIFFERENT AND MAY NEED EXTRA SET 

class DigitDetect:



    # screenshot() returns an image of some given area
    def screenshot(self, x, y, width, height
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

    # returns digit and position
    def detect(self, img, copy, region):
        tmp_numbers = {}
        
        # Read the input image 
        im = img

        # Convert to grayscale and apply Gaussian filtering; was COLOR_BGR2GRAY
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('images/numbers/'+ region +'/' + str(copy) + '.jpg', 0)
        w, h = template.shape[::-1]
        
        res = cv2.matchTemplate(im_gray, template, cv2.TM_CCOEFF_NORMED)
        
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

    def get_data(self, x_coor, y_coor, w, h, where, last = ''):
        start_1 = time.time()
        img = self.screenshot(x = x_coor, y = y_coor, width = w, height = h, gray = False)
        print(time.time()-start_1)
        numbers = {}
        for number in range(10):
            returned_numbers = self.detect(img = img, copy = number, region = where)
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
        if where == 'kda':
            kda = []
            for i in num_str:
                if i == '':
                    return last
                else:
                    kda.append(i)
            return kda
        if num_str == '':
            return last
        else:
            return num_str


# Following attempts to read and interpret on-screen information

for i in list(range(6))[::-1]:
    print(i+1)
    time.sleep(1)

# start = time.time()
# count = 0
# while True:
#     if count == 0:
#         print('Time elapsed:', time.time()-start)
#         print('Coins: ', DigitDetect().get_data(1117,868,51,17, where = 'coins'))
#         print('CP: ', DigitDetect().get_data(1504,173,25,16, where = 'cp'))
#         print('KDA: ', DigitDetect().get_data(1427,172,43,13, where = 'kda'))
#         print('-------------')
#         last_coins = DigitDetect().get_data(1117,868,51,17, where = 'coins')
#         last_cp = DigitDetect().get_data(1504,173,25,16, where = 'cp')
#         last_kda = DigitDetect().get_data(1427,172,43,13, where = 'kda')

#         count += 1
#     else:
#         print('Time elapsed:', time.time()-start)
        
#         print('Coins: ', DigitDetect().get_data(1117,868,51,17, where = 'coins', last = last_coins))
#         print('CP: ', DigitDetect().get_data(1504,173,25,16, where = 'cp', last = last_cp))
#         print('KDA: ', DigitDetect().get_data(1427,172,43,13, where = 'kda', last = last_kda))
#         print('-------------')
#         last_coins = DigitDetect().get_data(1117,868,51,17, where = 'coins')
#         last_cp = DigitDetect().get_data(1504,173,25,16, where = 'cp')
#         last_kda = DigitDetect().get_data(1427,172,43,13, where = 'kda')


#     time.sleep(5)

start = time.time()
count = 0
while True:
    if count == 0:
        print('Time elapsed:', time.time()-start)
        coins = DigitDetect().get_data(1117,868,51,17, where = 'coins')
        cp = DigitDetect().get_data(1504,173,25,16, where = 'cp')
        kda = DigitDetect().get_data(1427,172,43,13, where = 'kda')
        print('-------------')
        last_coins = coins
        last_cp = cp
        last_kda = kda

        count += 1
    else:
        print('Time elapsed:', time.time()-start)
        coins = DigitDetect().get_data(1117,868,51,17, where = 'coins', last = last_coins)
        cp = DigitDetect().get_data(1504,173,25,16, where = 'cp', last = last_cp)
        kda = DigitDetect().get_data(1427,172,43,13, where = 'kda', last = last_kda)
        print('-------------')
        last_coins = coins
        last_cp = cp
        last_kda = kda
        
    time.sleep(5)


# MAKE SCREENSHOT FASTER USE DIRECT WINDOWS METHOD!!!! get each screengrab down to 0.01 then we can total 
# time with each detection with 0.05 = 20fps
