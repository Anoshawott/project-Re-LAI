from keras.models import sequential
from keras.layers import Dense, Dropout, Conv2d, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard

class DQNAgent:
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))