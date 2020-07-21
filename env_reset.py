from window_cap import WindowCapture
from RealRecognition import Detection
import time
import subprocess
from pynput.mouse import Button, Controller
from DirectKeys import PressKey, ReleaseKey, START, ESC, Y

class EnvReset:
    def game_reset(self):
        mouse = Controller()

        # Exiting Game
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)

        try:
            WindowCapture('League of Legends (TM) Client')
            PressKey(ESC)
            ReleaseKey(ESC)
            time.sleep(1)

            mouse.position = (606, 841)
            mouse.click(Button.left, 2)
            time.sleep(1)

            mouse.position = (842, 537)
            mouse.click(Button.left, 1)
            time.sleep(1)
            time.sleep(20)
        except:
            None

        full_check = True
        while full_check:
            subprocess.call("taskkill /f /im \"LeagueClient.exe\"", shell=True)
            time.sleep(5)
            client_check = True
            while client_check:
                try:
                    WindowCapture('League of Legends')
                    client_check = False
                except:
                    mouse.position = (21, 1062)
                    time.sleep(0.01)
                    mouse.click(Button.left, 1)
                    mouse.position = (385, 643)
                    time.sleep(1)
                    mouse.click(Button.left, 1)
                    mouse.scroll(0,-100)
                    time.sleep(1.5)
                    mouse.position = (319, 832)
                    time.sleep(0.01)
                    mouse.click(Button.left, 1)
                    time.sleep(8)

            check = True
            while check:
                try:
                    WindowCapture('League of Legends')
                    check = False
                except:
                    None

            time.sleep(25)

            check = True
            while check:
                try:
                    WindowCapture('League of Legends (TM) Client')
                    subprocess.call("taskkill /f /im \"League of Legends.exe\"", shell=True)
                    check = False
                    full_check = True
                except:
                    full_check = False
                    check = False

        mouse.position = (549, 269)
        time.sleep(0.01)
        mouse.click(Button.left, 1)
        time.sleep(1)
        mouse.position = (773, 310)
        time.sleep(0.01)
        mouse.click(Button.left, 1)
        time.sleep(1)
        mouse.position = (898, 721)
        mouse.click(Button.left, 1)
        time.sleep(1)
        mouse.position = (872, 778)
        time.sleep(0.01)
        mouse.click(Button.left, 1)
        time.sleep(1)

        # Adding Ally Bots
        mouse.position = (810, 460)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (809, 499)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (814, 546)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (805, 579)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        # Adding Enemy Bots

        mouse.position = (1205, 425)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (1200, 463)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (1201, 501)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (1196, 540)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        mouse.position = (1198, 579)
        time.sleep(0.01)
        mouse.click(Button.left, 1)

        time.sleep(1)

        # Awaiting Game Start
        mouse.position = (874, 776)
        mouse.click(Button.left, 1)
        time.sleep(3)

        # Champion select
        mouse.position = (837, 363)
        mouse.click(Button.left, 1)
        time.sleep(1)

        mouse.position = (964, 719)
        mouse.click(Button.left, 1)
        time.sleep(50)

        # Getting start items
        mouse.position = (964, 719)
        mouse.click(Button.left, 1)
        time.sleep(2)
        mouse.position = (837, 363)
        mouse.click(Button.left, 1)
        time.sleep(2)
        mouse.position = (768, 282)
        mouse.click(Button.left, 2)
        time.sleep(1)

        mouse.position = (526, 247)
        mouse.click(Button.left, 1)
        mouse.click(Button.right, 1)
        time.sleep(1)

        mouse.position = (580, 247)
        mouse.click(Button.left, 1)
        mouse.click(Button.right, 1)
        time.sleep(1)
        mouse.click(Button.right, 1)
        time.sleep(1)

        mouse.position = (626, 247)
        mouse.click(Button.left, 1)
        mouse.click(Button.right, 1)
        time.sleep(1)

        PressKey(ESC)
        ReleaseKey(ESC)
        time.sleep(1)

        PressKey(Y)
        ReleaseKey(Y)
        time.sleep(1)

        # Getting in position
        mouse.position = (1492, 810)
        mouse.click(Button.right, 1)
        time.sleep(35)

        print('Done!')