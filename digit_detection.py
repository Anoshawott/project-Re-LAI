import numpy as np
import cv2
import mss
import mss.tools
    
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
        img = self.screenshot(x = x_coor, y = y_coor, width = w, height = h, gray = False)
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
            return int(num_str)