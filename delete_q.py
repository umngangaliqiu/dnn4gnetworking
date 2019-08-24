from utils import *
import numpy as np
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import optimizers
from matplotlib import pyplot as plt
import scipy.io as sio
from networkmpc import mpc
from scipy.stats import truncnorm
import tensorflow as tf
import copy
# SEED=1234
# np.random.seed(SEED)
a = np.array((1,2,3,4))
print(a)
print(a+1)