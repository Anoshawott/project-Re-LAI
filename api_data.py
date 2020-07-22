import urllib.request, json 
import time
import requests

def realtime_capture(wincap=None):
    loop_time = time.time()
    while(True):
        
        # screenshot = wincap.get_screenshot()
        
        r = requests.get("https://127.0.0.1:2999/liveclientdata/activeplayer", verify=False)
        data = r.json()
        print(data)

        # debugging the loop rate
        print('FPS {}'.format(1/(time.time()-loop_time)))
        loop_time = time.time()

    return print('Done.')

realtime_capture()