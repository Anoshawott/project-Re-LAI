import urllib.request, json 
import time
import requests
import copy
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

### FOR SOME REASON SCORES (I.E. KDA AND CS) IS NOT BEING RECORDED PROPERLY??? WHY?? 

def realtime_capture(wincap=None):
    curr_dict = {'currentHealth': 541.0, 'resourceValue': 418.0, 
                    'level': 1, 'assists': 0, 'creepScore': 0, 
                    'deaths': 0, 'kills': 0, 'wardScore': 0.0}
    loop_time = time.time()
    while(True):
    
        # screenshot = wincap.get_screenshot()
        
        r = requests.get("https://127.0.0.1:2999/liveclientdata/allgamedata", verify=False)
        data = r.json()
        thing = ['currentHealth', 'resourceValue']
        new_dict = {}
        for i in thing:
            new_dict[i] = data['activePlayer']['championStats'][i]
        new_dict['level'] = data['activePlayer']['level']
        # print(new_dict)
        # print(len(data['allPlayers']))
        # print(data['allPlayers']['championName'])
        for i in range(0,len(data['allPlayers'])):
            if data['allPlayers'][i]['championName'] == 'Ahri':
                print(data['allPlayers'][i]['scores'])
                thing2 = ['assists', 'creepScore', 'deaths', 'kills', 'wardScore']
                for k in thing2:
                    new_dict[k] = data['allPlayers'][i]['scores'][k]

        output_data_comp = curr_dict.items() & new_dict.items()
        if len(output_data_comp) != len(curr_dict):
            print(new_dict)
        
        curr_dict = copy.deepcopy(new_dict)

        # debugging the loop rate
        print('FPS {}'.format(1/(time.time()-loop_time)))
        loop_time = time.time()

    return print('Done.')

realtime_capture()