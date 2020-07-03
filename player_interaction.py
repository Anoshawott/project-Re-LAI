from detection import Detection

class PlayerAI:
    def __init__(self):
        self.x_map, self.y_map = Detection().detect(Detection().screenshot(),1087,527,181,181,player=True)
        self.x, self.y = ''
    
    def action(self, action):
        # each choice should be associated with a chosen position of the 
        return