from window_cap import WindowCapture
from detection import Detection
import cv2
import time

# wincap = WindowCapture('screenshots')
# img = wincap.get_screenshot()
# print(wincap.get_screen_position((0,0)))
# cv2.imshow('img', img)
# cv2.waitKey()

def realtime_capture(wincap=None):
    # loop_time = time.time()
    while(True):
        
        # screenshot = wincap.get_screenshot()
        
        x_pos, y_pos, img = Detection().detect(Detection().screenshot(),
                                        x = 1087,y = 527, width = 181, height = 181,player=True)
        print(x_pos, y_pos)
        cv2.imshow('player identified?', img)

        # debugging the loop rate
        # print('FPS {}'.format(1/(time.time()-loop_time)))
        # loop_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    return print('Done.')

realtime_capture()


