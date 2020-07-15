from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf

model_1 = tf.keras.models.load_model('dql_models_per_5/12X2__ep___20.00_-200.00max_-4870.00avg_-9540.00min__1594721228.model')
model_2 = tf.keras.models.load_model('models/2x256___125.00ep___24.00max_-190.40avg_-497.00min__1594777317.model')
model_3 = tf.keras.models.load_model('models/2x256____25.00ep_-200.00max_-274.50avg_-349.00min__1594776816.model')

print('-----------MY IMPLEMENTATION-----------')
print(model_1.summary())

print('-----------EXAMPLE-----------')
print(model_2.summary())
print(model_3.summary())