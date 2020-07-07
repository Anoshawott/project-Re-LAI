from DirectKeys import PressKey, ReleaseKey, Y, Q, W, E, R, D, F, B, ESC, CONTROL, num_1, num_2, num_3, num_4, num_5, num_6 
import pickle

class ChoiceCreate:
    def choices_dict(self):
        choices = {}

        # replace keyboard inputs with variables from DirectKeys
        radii = [100,200,400,550,700]
        keyboard_actions = ['', Q, W, E, R, D, F]
        abilities = [Q, W, E, R]
        items = [num_1,num_2,num_3,num_4,num_5,num_6]

        # Movement and ability actions
        count=0
        for j in radii:
            for k in range(1,37):
                for i in keyboard_actions:
                    choices[count]=[j,k*10,i]
                    count+=1

        # Level-up actions
        for i in abilities:
            choices[count]=[CONTROL, i]
            count+=1

        # Item actions
        for i in items:
            choices[count]=[i]
            count+=1
        return choices
        
    def choices_save(self):
        choices = self.choices_dict()
        pickle_out = open('choices.pickle','wb')
        pickle.dump(choices, pickle_out)
        pickle_out.close()
        return print('Done!')

ChoiceCreate().choices_save()
