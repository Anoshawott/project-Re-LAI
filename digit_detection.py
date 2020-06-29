import numpy as np
import cv2
import mss
import mss.tools
    
##### SUCCESS!!!!!!!
# NOTE: 0.8 = get_coins, GET_HP WILL NEED ITS OWN SET OF NUMBERS TO COMPARE, ALSO NUMBERS FOR
# METRICS MAY BE DIFFERENT AND MAY NEED EXTRA SET 

class DigitDetect:

    # screenshot() returns an image of some given area
    @staticmethod
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

    # returns digit and position
    @staticmethod
    def detect(img, copy, region):
        tmp_numbers = {}
        
        # Read the input image 
        im = cv2.imread(img)

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
                tmp_numbers[str(copy)+'_1'] = pt
                count += 1
            elif count == 2:
                tmp_numbers[str(copy)+'_2'] = pt
                count += 1
            elif count == 3:
                tmp_numbers[str(copy)+'_3'] = pt
            else:
                tmp_numbers[str(copy)] = pt
                count += 1

        print('Yesss!')
        return tmp_numbers

    @staticmethod
    def get_coins():
        img = screenshot(975, 718, 50, 20, gray = False)
        numbers = {}
        for number in range(10):
            returned_numbers = detect(img, number, region = 'coins')
            z = numbers.copy()
            z.update(returned_numbers)
            numbers = z
        num_coins = {k: v for k, v in sorted(numbers.items(), key=lambda item: item[1])}
        return num_coins

