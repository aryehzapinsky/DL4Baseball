
# coding: utf-8

# # Deep Learning for Computer Vision:  Name / No name classifier

# ## Aryeh Zapinsky and Jonathan Herman

# In[116]:


import os
import shutil
import h5py

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend
from keras import optimizers

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Prepare data
# 
# ##### Get data into this form:

# In[2]:


# DL4Baseball
# |-- name_data`
# |   |-- train
# |       |-- name
# |       |-- not_name
# |---|-- validation
#         |-- name
#         |-- not_name
# ...


# In[4]:


# Separate data into at-bats, not at-bats, training and validation
get_ipython().run_line_magic('mkdir', '-p ./name_data/train/name')
get_ipython().run_line_magic('mkdir', '-p ./name_data/train/no_name')
get_ipython().run_line_magic('mkdir', '-p ./name_data/validation/name')
get_ipython().run_line_magic('mkdir', '-p ./name_data/validation/no_name')


# In[ ]:


#### Count files 
#find ./name/ -type f | wc -l         # 580
#find ./no_name/ -type f | wc -l     # 13,817


# ### Fill training and validation dirs
# 
# ##### Using 87/13 train/test split:
# 
# ./name_data/train/name -- 500 samples
# 
# ./name_data/train/no_name -- 11912 samples
# 
# ./name_data/validation/name -- 80 samples
# 
# ./name_data/validation/no_name -- 1905 samples
# 
# We will augment the data to have even more samples.

# In[9]:


def prepare_dir(label, dest, size):
    '''Move SIZE samples of LABEL imgs into DEST'''
    source = './name_data/' + label + '/'
    dest = './name_data/' + dest + '/' + label + '/' 
    # random selection of SIZE of LABEL
    train_sample = np.random.choice(os.listdir(source), size=size, replace=False)
    
    for fp in train_sample:
        shutil.move(source + fp, dest + fp)        

# Prepare training sets according to sizes above
print('Preparing name training set...')
prepare_dir('name', dest='train', size=500)
print('Preparing no_names training set...')
prepare_dir('no_name', dest='train', size=11912)

# Prepare validation sets according to sizes above
print('Preparing name validation set...')
prepare_dir('name', dest='validation', size=80)
print('Preparing no_name validation set...')
prepare_dir('no_name', dest='validation', size=1905)


# ### Augment Data

# In[22]:


# Image info
train_data_dir = './name_data/train/'
validation_data_dir = './name_data/validation/'
img_width, img_height = (300, 300)

# Calculate steps per epoch
nb_train_samples = 12411
nb_validation_samples = 1984
batch_size = 32
steps_per_epoch_train = nb_train_samples / batch_size
steps_per_epoch_val = nb_validation_samples / batch_size


# In[86]:


# Data augmentation for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.3,
        height_shift_range=0.2,
        fill_mode='nearest')

# Data augmentation for testing (only scaling)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


# Let's display some sample images from this dataset. This is like finding a needle in a haystack with 1/20 shot of finding a name image.

# In[62]:


for X_batch, Y_batch in validation_generator:
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X_batch[i])
        plt.title("C = {}".format(Y_batch[i]))
        plt.axis('off')
    plt.show()
    break


# ## 1. Build VGG-16

# In[76]:


def build_vgg16(framework='tf'):

    if framework == 'th':
        # build the VGG16 network in Theano weight ordering mode
        backend.set_image_dim_ordering('th')
    else:
        # build the VGG16 network in Tensorflow weight ordering mode
        backend.set_image_dim_ordering('tf')
        
    model = Sequential()
    if framework == 'th':
        model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3)))
        
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    return model


# ## 2. Load pretrained weights

# Now we build the model using tensorflow format and load the weights.

# In[77]:


# path to the model weights files.
weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
tf_model = build_vgg16('tf')
tf_model.load_weights(weights_path)


# Next we make the last layer or layers. We flatten the output from the last convolutional layer, and add fully connected layer with 256 hidden units. Finally, we add the output layer which is a scalar output as we have a binary classifier. 

# ## 3. Add a fully connected layer

# In[78]:


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
print (Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Flatten(input_shape=tf_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
print (tf_model.summary())
print(top_model.summary())


# We add this model to the top of our VGG16 network, freeze all the weights except the top, and compile.

# In[79]:


# add the model on top of the convolutional base
tf_model.add(top_model)


# ## 4. Freeze all but last two layers

# In[80]:


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in tf_model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
tf_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# ## 5. Fine tune network on Dogs and Cats

# Now we train for 5 epochs to get the weights for the top close to where we need them. Essentially, we want the network to be doing the right thing before we unnfreeze the lower weights.

# In[81]:


# Bookkeeping
get_ipython().run_line_magic('mkdir', '-p ./logs/vgg16_top_tuning')
get_ipython().run_line_magic('mkdir', '-p ./history')
get_ipython().run_line_magic('mkdir', '-p ./models')


# In[82]:


nb_epochs = 5

tensorboard_callback = TensorBoard(log_dir='./logs/vgg16_top_tuning/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/vgg16_top_tuning_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      mode='auto', period=1)

vgg16_top_convet = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, checkpoint_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(vgg16_top_convet.history).to_csv("./history/vgg16_top_tuning_weights.csv")


# In[85]:


tf_model = load_model('./models/vgg16_top_tuning_best.hdf5')

nb_epochs = 7

tensorboard_callback = TensorBoard(log_dir='./logs/vgg16_top_tuning/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/vgg16_top_tuning_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      mode='auto', period=1)

vgg16_top_convet = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, checkpoint_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(vgg16_top_convet.history).to_csv("./history/vgg16_top_tuning_weights.csv")


# # THIS IS NOT WORKING AS IT SHOULD

# In[87]:


tf_model = load_model('./models/vgg16_top_tuning_best.hdf5')

nb_epoch = 10
vgg16_top_convet = tf_model.fit_generator(train_generator, 
              initial_epoch=7, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, checkpoint_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(vgg16_top_convet.history).to_csv("./history/vgg16_top_tuning_weights.csv")


# ## 6. Evaluate Accuracy

# Running this, we see that it gets 92% accuracy on the validation set, so we've halved the errors from before.

# In[98]:


accuracies = np.array([])
losses = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = tf_model.evaluate(X_batch, Y_batch, verbose=0)
    losses = np.append(losses, loss)
    accuracies = np.append(accuracies, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation: accuracy = %f  ;  loss = %f" % (np.mean(accuracies),
                                                   np.mean(losses)))


# In[94]:


nb_classes = 2
class_name = {
    0: 'name',
    1: 'no_name',
}

def show_sample(X, y, prediction=-1):
    im = X
    plt.imshow(im)
    if prediction >= 0:
        plt.title("Class = %s, Predict = %s" % (class_name[y], class_name[prediction]))
    else:
        plt.title("Class = %s" % (class_name[y]))

    plt.axis('off')
    plt.show()


# In[99]:


X_test, y_test = next(validation_generator)
predictions = tf_model.predict_classes(X_test, batch_size=32, verbose=0)

for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions[i, 0])


# ## 7. Unfreeze all layers

# Now we can unnfreeze the lower layers.

# In[100]:


# set all layers to trainable (updating weights)
for layer in tf_model.layers[:25]:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
tf_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# ## 8. Fine tune entire network

# We will let this train for 10 epochs.

# In[101]:


tf_model = load_model('./models/vgg16_top_tuning_best.hdf5')

nb_epochs = 10

tensorboard_callback = TensorBoard(log_dir='./logs/vgg16_whole_after_top/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/vgg16_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      mode='auto', period=1)

vgg16_whole_convet = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, checkpoint_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(vgg16_whole_convet.history).to_csv("./history/vgg16_whole_after_top_weights.csv")


# In[117]:


tf_model = load_model('./models/vgg16_top_tuning_best.hdf5')

# Data augmentation for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.3,
        height_shift_range=0.2,
        fill_mode='nearest')

# Data augmentation for testing (only scaling)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

nb_epochs = 10

tensorboard_callback = TensorBoard(log_dir='./logs/vgg16_whole_early_stop/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/vgg16_early_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                        mode='auto', period=1)

earlystop_callback = EarlyStopping(monitor='val_loss', 
                                   min_delta=0.01, 
                                   patience=10,
                                   verbose=0, 
                                   mode='auto')



vgg16_whole_convet = tf_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, 
                         checkpoint_callback,
                        earlystop_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(vgg16_whole_convet.history).to_csv("./history/vgg16_whole_early_stop_weights.csv")


# # 9. Evaluate accuracy

# We get to 97% accuracy!

# In[119]:


accuracies = np.array([])
losses = np.array([])

i=0
for X_batch, Y_batch in validation_generator:
    loss, accuracy = tf_model.evaluate(X_batch, Y_batch, verbose=0)
    losses = np.append(losses, loss)
    accuracies = np.append(accuracies, accuracy)
    i += 1
    if i == 20:
       break
       
print("Validation: accuracy = %f  ;  loss = %f" % (np.mean(accuracies),
                                                   np.mean(losses)))


# In[ ]:


X_test, y_test = next(validation_generator)
predictions = tf_model.predict_classes(X_test, batch_size=32, verbose=0)

for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions[i, 0])


# 
# 
# 
# 
# # !!!! REFINING NAME CLASSIFIER !!!!

# In[1]:


import os
import shutil
import h5py

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import backend
from keras import optimizers

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Prepare data
# 
# ##### Get data into this form:

# In[2]:


# DL4Baseball
# |-- name_data`
# |   |-- train
# |       |-- name
# |       |-- not_name
# |---|-- validation
#         |-- name
#         |-- not_name
# ...


# ### Augment Data

# In[2]:


# Image info
train_data_dir = './refined_namesset/train/'
validation_data_dir = './refined_namesset/validation/'
img_width, img_height = (300, 300)


# In[9]:


batch_size = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest')

# Data augmentation for testing (only scaling)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


# In[4]:


# Calculate steps per epoch
nb_train_samples = 1277
nb_validation_samples = 230
steps_per_epoch_train = nb_train_samples / batch_size
steps_per_epoch_val = nb_validation_samples / batch_size


# ### Load Model

# In[8]:


model = load_model('./models/vgg16_best.hdf5')


# ### Freeze all but last two layers

# In[9]:


# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[10]:


nb_epochs = 5

tensorboard_callback = TensorBoard(log_dir='./logs/namenet_initial_tuning/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/namenet_initial_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      mode='auto', period=1)

top_convet = model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, checkpoint_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(top_convet.history).to_csv("./history/namenet_initial_weights.csv")


# ### Unfreeze all layers

# In[5]:


whole_model = load_model('./models/namenet_initial_best.hdf5')

# set all layers to trainable (updating weights)
for layer in whole_model.layers[:25]:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
whole_model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# In[6]:


nb_epochs = 4

tensorboard_callback = TensorBoard(log_dir='./logs/namenet_fine_tuning/', 
                                   histogram_freq=0, 
                                   write_graph=True, 
                                   write_images=False)
checkpoint_callback = ModelCheckpoint('./models/namenet_entire_best.hdf5', 
                                      monitor='val_acc', 
                                      verbose=0, 
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      mode='auto', period=1)

earlystop_callback = EarlyStopping(monitor='val_loss', 
                                   min_delta=0.01, 
                                   patience=10,
                                   verbose=0, 
                                   mode='auto')

whole_convet = whole_model.fit_generator(train_generator, 
              initial_epoch=0, 
              verbose=1, 
              validation_data=validation_generator, 
              steps_per_epoch=steps_per_epoch_train, 
              epochs=nb_epochs, 
              callbacks=[tensorboard_callback, 
                         checkpoint_callback, 
                         earlystop_callback],
              validation_steps=steps_per_epoch_val)

pandas.DataFrame(whole_convet.history).to_csv("./history/namenet_fine_weights.csv")


# ## Look at results

# In[8]:


nb_classes = 2
class_name = {
    0: 'name',
    1: 'no_name',
}

def show_sample(X, y, prediction=-1):
    im = X
    plt.imshow(im)
    if prediction >= 0:
        plt.title("Class = %s, Predict = %s" % (class_name[y], class_name[prediction]))
    else:
        plt.title("Class = %s" % (class_name[y]))

    plt.axis('off')
    plt.show()


# In[11]:


X_test, y_test = next(validation_generator)
predictions = whole_model.predict_classes(X_test, batch_size=32, verbose=0)

for i in range(32):
    show_sample(X_test[i, :, :, :], y_test[i], prediction=predictions[i, 0])

