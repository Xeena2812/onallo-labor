# %% [markdown]
# ## Importing the CIFAR-100 dataset

# %%
import numpy as np
import tensorflow as tf

# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelBinarizer

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')

print(x_train.shape, y_train.shape)

# %%
print(tf.config.list_physical_devices('GPU'))

# %%
x_train = (x_train / 255 - 0.5) * 2
x_test = (x_test / 255 - 0.5) * 2

enc = LabelBinarizer()

y_train = enc.fit_transform(y_train)
y_test = enc.fit_transform(y_test)


# %%
print(x_test[0].shape)

# %% [markdown]
# ## Inception v4 Implementation
# 
# The CIFAR100 dataset is composed of 32x32 pixel images as oppopsed to the 299x299 pixel images of the Imagenet network for which the Inceptionv4 network was originally comstructed. There a significant downsizing of the network is necessary to accomodate the 10x size difference.
# 
# Inception blocks are used to create feature maps, and reduction blocks are used to downsize the filter size for teh inception blocks
# Intuitions:
# 1. Put 5x5 convolutions in stem to downsize image to smaller size, and expand filter space for the Inception-A blocks.
# 2. First inception blocks TRANSFORM the feature map (with 1x1 convolutions) while KEEPING the same image size.
# 3. Reduction-A blocks further reduce the image size and expand filter space for Inception-B blocks.
# 4. Inception-B blocks further transform the feature map.
# 5. Probably not neccessary, but if it is add Reduction-B blocks
# 6. A final AveragePooling layer reduces  
# 
# 1. Drop the reduction blocks as the image is already small adn there is no need to reduce image size (probably only dimension reduction is needed)
#    1. this probably won't work as different size feature extraction is necessary
# 2. Reduce the number of Conv2DBatchNorm layers in each block (Inception, Reduction, Stem) in order not to decrease the image size down to one so quickly
# 3. Change all filter numbers to half.

# %%
#import tensorflow as tf
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D, Dropout, Concatenate, Flatten
#from keras.layers import *
from keras import Model


# TODO: Go to Inceptionv2 paper to understand Conv2D_BN and fine tune parameters potentially
def Conv2DBatchNorm(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
            #    kernel_regularizer=keras.regularizers.l2(0.00004),
            #    kernel_initializer=keras.initializers.VarianceScaling(scale=0.2, mode='fan_in', distribution='normal', seed=None)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(x.shape)
    return x

def Stem(x):
    print("pre-stem size:", x.shape)

    x = Conv2DBatchNorm(x, 16, (5, 5))
    x = Conv2DBatchNorm(x, 32, (3, 3))

    branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    branch_1 = Conv2DBatchNorm(x, 48, (3, 3), strides=(2, 2), padding='valid')

    x = Concatenate()([branch_0, branch_1])
    print("Concat 1")
    branch_0 = Conv2DBatchNorm(x, 32, (1, 1))
    branch_0 = Conv2DBatchNorm(branch_0, 48, (3, 3), padding='valid')

    branch_1 = Conv2DBatchNorm(x, 32, (1, 1))
    branch_1 = Conv2DBatchNorm(branch_1, 32, (1, 7))
    branch_1 = Conv2DBatchNorm(branch_1, 32, (7, 1))
    branch_1 = Conv2DBatchNorm(branch_1, 48, (3, 3), padding='valid')

    x = Concatenate()([branch_0, branch_1])
    print("Concat 2")
    branch_0 = Conv2DBatchNorm(x, 96, (3, 3), strides=(2, 2), padding='valid')
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = Concatenate()([branch_0, branch_1])

    print("post-stem size:", x.shape)

    return x


def InceptionA(x):
    print("pre-A size:", x.shape)

    branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_0 = Conv2DBatchNorm(branch_0, 96, (1, 1))

    branch_1 = Conv2DBatchNorm(x, 96, (1, 1))

    branch_2 = Conv2DBatchNorm(x, 64, (1, 1))
    branch_2 = Conv2DBatchNorm(branch_2, 96, (3, 3))

    branch_3 = Conv2DBatchNorm(x, 64, (1, 1))
    branch_3 = Conv2DBatchNorm(branch_3, 96, (3, 3))
    branch_3 = Conv2DBatchNorm(branch_3, 96, (3, 3))

    x = Concatenate()([branch_0, branch_1, branch_2, branch_3])
    print("post-A size:", x.shape)

    return x


def InceptionB(x):
    print("pre-B size:", x.shape)

    branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_0 = Conv2DBatchNorm(branch_0, 128, (1, 1))

    branch_1 = Conv2DBatchNorm(x, 384, (1, 1))

    branch_2 = Conv2DBatchNorm(x, 192, (1, 1))
    branch_2 = Conv2DBatchNorm(branch_2, 224, (1, 7))
    branch_2 = Conv2DBatchNorm(branch_2, 256, (7, 1))

    branch_3 = Conv2DBatchNorm(x, 192, (1, 1))
    branch_3 = Conv2DBatchNorm(branch_3, 192, (1, 7))
    branch_3 = Conv2DBatchNorm(branch_3, 224, (7, 1))
    branch_3 = Conv2DBatchNorm(branch_3, 224, (1, 7))
    branch_3 = Conv2DBatchNorm(branch_3, 256, (7, 1))

    x = Concatenate()([branch_0, branch_1, branch_2, branch_3])
    print("post-B size:", x.shape)

    return x


def InceptionC(x):
    print("pre-C size:", x.shape)

    branch_0 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_0 = Conv2DBatchNorm(branch_0, 256, (1, 1))

    branch_1 = Conv2DBatchNorm(x, 256, (1, 1))

    branch_2 = Conv2DBatchNorm(x, 384, (1, 1))
    branch_2_0 = Conv2DBatchNorm(branch_2, 256, (1, 3))
    branch_2_1 = Conv2DBatchNorm(branch_2, 256, (3, 1))

    branch_3 = Conv2DBatchNorm(x, 384, (1, 1))
    branch_3 = Conv2DBatchNorm(branch_3, 448, (1, 3))
    branch_3 = Conv2DBatchNorm(branch_3, 512, (3, 1))
    branch_3_0 = Conv2DBatchNorm(branch_3, 256, (3, 1))
    branch_3_1 = Conv2DBatchNorm(branch_3, 256, (1, 3))

    x = Concatenate()([branch_0, branch_1, branch_2_0, branch_2_1, branch_3_0, branch_3_1])
    print("post-C size:", x.shape)

    return x


def ReductionA(x):
    print("pre-RedA size:", x.shape)

    branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    branch_1 = Conv2DBatchNorm(x, 384, (3, 3), strides=(2, 2), padding='valid')

    branch_2 = Conv2DBatchNorm(x, 192, (1, 1))
    branch_2 = Conv2DBatchNorm(branch_2, 224, (3, 3))
    branch_2 = Conv2DBatchNorm(branch_2, 256, (3, 3), strides=(2, 2), padding='valid')

    x = Concatenate()([branch_0, branch_1, branch_2])
    print("post-RedA size:", x.shape)

    return x


def ReductionB(x):
    print("pre-RedB size:", x.shape)

    branch_0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    branch_1 = Conv2DBatchNorm(x, 192, (1, 1))
    branch_1 = Conv2DBatchNorm(branch_1, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_2 = Conv2DBatchNorm(x, 256, (1, 1))
    branch_2 = Conv2DBatchNorm(branch_2, 256, (1, 7))
    branch_2 = Conv2DBatchNorm(branch_2, 320, (7, 1))
    branch_2 = Conv2DBatchNorm(branch_2, 320, (3, 3), strides=(2, 2), padding='valid')

    x = Concatenate()([branch_0, branch_1, branch_2])
    print("post-RedB size:", x.shape)

    return x


def MyInception():
    inputs = Input(shape=(32, 32, 3), name='input')

    x = Stem(inputs)

    for _ in range(4):
        x = InceptionA(x)

    x = ReductionA(x)

    for _ in range(7):
        x = InceptionB(x)

    x = ReductionB(x)

    for _ in range(3):
        x = InceptionC(x)

    x = AveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    outputs = Dense(100, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=outputs, name='MyInception')

    return model


model = MyInception()

model.compile()

model.summary()

# %%



