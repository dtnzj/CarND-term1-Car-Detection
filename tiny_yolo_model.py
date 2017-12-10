from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


def tiny_yolo():
    model = Sequential()
    """
    [convolutional]
    batch_normalize=1
    filters=16
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2
    stride=2
    """
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    """
    [convolutional]
    batch_normalize=1
    filters=32
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(32,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2
    stride=2
    """
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    """
    [convolutional]
    batch_normalize=1
    filters=64
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(64,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2
    stride=2
    """
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    """
    [convolutional]
    batch_normalize=1
    filters=128
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(128,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2    
    stride=2
    """
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    """
    [convolutional]
    batch_normalize=1
    filters=256
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(256,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2
    stride=2
    """
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    """
    [convolutional]
    batch_normalize=1
    filters=512
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(512,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [maxpool]
    size=2
    stride=1
    """
    model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
    
    """
    [convolutional]
    batch_normalize=1
    filters=1024
    size=3
    stride=1
    pad=1
    activation=leaky
    """
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    ###########
    [convolutional]
    batch_normalize=1
    size=3
    stride=1
    pad=1
    filters=1024
    activation=leaky
    """
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    """
    [convolutional]
    size=1
    stride=1
    pad=1
    filters=125
    activation=linear
    """ 
    model.add(Convolution2D(1024,3,3 ,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    
    # -------------------------------------------------------
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(1470))
    return model