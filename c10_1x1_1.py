import time
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers.merge import Add
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras.initializers import VarianceScaling

# HYPERPARAMETERS
d     = 2       # number of times each block is repeated
w     = 42      # width of convolutional layers
drop  = 0.1     # dropout
decay = 0.0001  # l2 regularization (weight decay)
scale = 1.0     # scale for weight initialization
epoch = 150     # number of epochs
batch = 125     # batch size
val   = False    # whether or not to split the training set in train/val.

init = VarianceScaling(scale=scale)

def block(width,x,stride=1,name=None):
    """Implementation of the B_1x1(1) block.

    Inputs:
        width (int) - width of convolutional layer.
        x (keras Layer) - previous layer of network.
        stride (int, optional) - spatial stride of convolutions, defaults to 1.
        name (str, optional) - name of the layer, defaults to None.

    Returns:
        x (keras Layer) - resulting convolutional block.
    """

    lin = Conv2D(width,1,kernel_initializer=init,kernel_regularizer=l2(decay),padding='same',strides=stride,name='{}_lin'.format(name))(x)
    act = Conv2D(width,3,kernel_initializer=init,kernel_regularizer=l2(decay),padding='same',activation='relu',strides=stride,name='{}_act'.format(name))(x)
    x = Add()([lin,act])
    x = Dropout(drop)(x)
    return x

def split(X,y,p=.2):
    """Create a train/val split of ratio (1-p):p.

    Inputs:
        X (ndarray) - array of images of shape (batch_size, width, height, channels).
        y (ndarray) - array of labels.
        p (float, optional) - percentage of training data to set aside as val.
    Returns:
        X_train (ndarray) - array of new training images.
        y_train (ndarray) - array of new training labels.
        X_val (ndarray) - array of new val images.
        y_val (ndarray) - array of new val labels.
    """
    out = []
    for i in np.unique(y):
        out.append(np.where(y==i)[0])
    train_idx = np.concatenate([o[:-int(o.size*p)] for o in out])
    val_idx = np.concatenate([o[-int(o.size*p):] for o in out])
    print train_idx.size, val_idx.size
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
if val: x_train, y_train, x_val, y_val = split(x_train, y_train)

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)
if val: y_val   = to_categorical(y_val, 10)

x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.
if val: x_val   = x_val.astype('float32') / 255.


input_img = Input(shape=(32, 32, 3))
x = input_img

x = Conv2D(1*w,1,kernel_initializer=init,kernel_regularizer=l2(decay),padding='same',name='conv1')(x)

layer_name_counter = 1
for _ in range(2*d):
    layer_name_counter += 1
    x = block(1*w,x,name='conv{}'.format(layer_name_counter))

layer_name_counter += 1
x = block(2*w,x,stride=2,name='conv{}'.format(layer_name_counter))
for _ in range(2*d-1):
    layer_name_counter += 1
    x = block(2*w,x,name='conv{}'.format(layer_name_counter))

layer_name_counter += 1
x = block(4*w,x,stride=2,name='conv{}'.format(layer_name_counter))
for _ in range(2*d-1):
    layer_name_counter += 1
    x = block(4*w,x,name='conv{}'.format(layer_name_counter))

x = GlobalAveragePooling2D()(x)
output_class = Dense(10, activation='softmax')(x)

model = Model(input_img, output_class)
print 'Parameters:',model.count_params()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def schedule(ep):
    if ep <= int(.6*epoch):
        return 0.001
    if ep <= int(.8*epoch):
        return 0.0002
    else:
        return 0.00004

lr_schedule = LearningRateScheduler(schedule)

datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)

if val:
    val_data = (x_val, y_val)
else:
    val_data = (x_test, y_test)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch),
                    steps_per_epoch=x_train.shape[0] // batch,
                    epochs=epoch,
                    callbacks=[lr_schedule],
                    validation_data=val_data,
                    workers=1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print "Test Acc:", test_acc
