import numpy as np
from keras.applications.vgg19 import VGG19


vgg19_model = VGG19(weights='imagenet', include_top=False)
vgg19_model.summary()
