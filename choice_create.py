from DirectKeys import PressKey, ReleaseKey, Y, Q, W, E, R, D, F, B, ESC, CONTROL, num_1, num_2, num_3, num_4, num_5, num_6 
import pickle

class ChoiceCreate:
    def choices_dict(self):
        choices = {}

        # replace keyboard inputs with variables from DirectKeys
        radii = [200,400]
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

    def min_tur_dict(self):
        return {'tur_outer':9999, 'tur_inner':9999, 'tur_inhib':9999, 
                'inhib':9999, 'tur_nex_1':9999, 'tur_nex_2':9999, 'nexus':9999}

    def min_tur_save(self):
        min_tur = self.min_tur_dict()
        pickle_out = open('min_tur.pickle','wb')
        pickle.dump(min_tur, pickle_out)
        pickle_out.close()
    
    def tur_status_dict(self):
        return {'tur_outer':1, 'tur_inner':0, 'tur_inhib':0, 
                'inhib':0, 'tur_nex_1':0, 'tur_nex_2':0, 'nexus':0}
    
    def tur_status_save(self):
        tur_status = self.tur_status_dict()
        pickle_out = open('tur_status.pickle','wb')
        pickle.dump(tur_status, pickle_out)
        pickle_out.close()

ChoiceCreate().choices_save()
