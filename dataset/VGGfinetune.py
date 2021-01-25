# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:49:49 2019

@author: ushasi2
help:
https://keras.io/applications/#resnet
"""
import matplotlib.pyplot as plt
#import tensorflow
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16
from tensorflow.python.keras.models import Sequential, Model
#from keras.preprocessing import image
from tensorflow.keras.layers import Dense,  Flatten#Activation,
from tensorflow.python.keras.layers import Dense, Dropout
import scipy.io as sio
import numpy as np
import os


NUM_CLASSES = 125		####
CHANNELS = 3
IMAGE_RESIZE = 224	####
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']
NUM_EPOCHS = 50

train_batchsize = 100
val_batchsize = 100
test_batchsize = 1

train_dir = 'photo/tx_000000000000'
validation_dir = 'photo/tx_000000000000'
test_dir = 'photo/tx_000000000000'

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3"


# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()








# Utility function for obtaining of the errors 
def obtain_errors(val_generator, predictions):
    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the dictionary of classes
    label2index = validation_generator.class_indices

    # Obtain the list of the classes
    idx2label = list(label2index.keys())
    print("The list of classes: ", idx2label)

    # Get the class index
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("Number of errors = {}/{}".format(len(errors),validation_generator.samples))
    
    return idx2label, errors, fnames


# Utility function for visualization of the errors
def show_errors(idx2label, errors, predictions, fnames):
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])

        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()

image_size = 224

vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze all the layers
for layer in vgg_conv.layers[:]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# Create the model
model = Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)
# Add new layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

#model.layers[0].trainable = True
# Show a summary of the model. Check the number of trainable parameters
model.summary()

# Load the normalized images
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM

# Data generator for training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')


# Data generator for validation data
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

# Configure the model for training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit(
      train_generator,
      steps_per_epoch=
         train_generator.samples/train_generator.batch_size,
      epochs=1,
      validation_data=validation_generator, 
      validation_steps=
         validation_generator.samples/validation_generator.batch_size,
      verbose=1)



# Get the predictions from the model using the generator
predictions = model.predict(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
'''
# Run the function to illustrate accuracy and loss
visualize_results(history)


# Run the function to get the list of classes and errors
idx2label, errors, fnames = obtain_errors(validation_generator, predictions)

# Run the function to illustrate the error cases
show_errors(idx2label, errors, predictions, fnames)

'''
predictions = model.predict(validation_generator,verbose=1)



# NOTE that flow_from_directory treats each sub-folder as a class which works fine for training data
# Actually class_mode=None is a kind of workaround for test data which too must be kept in a subfolder

# batch_size can be 1 or any factor of test dataset size to ensure that test dataset is samples just once, i.e., no data is left out

test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (image_size, image_size),
    batch_size = test_batchsize,
    class_mode = None,
    shuffle = False,
    seed = 123
)

#Need to compile layer[0] for extracting the 256- dim features.
#model.layers[1].compile(optimizer = sgd, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)

test_generator.reset()
#pred = model.layers[1].predict_generator(test_generator, steps = len(test_generator), verbose = 1) 
f =  Model(inputs=model.input,outputs=model.get_layer('dense').output)
f.compile(optimizer = optimizers.RMSprop(lr=1e-4), loss ='categorical_crossentropy', metrics = ['acc']) 
for idx,layer in enumerate(f.layers):
        if(layer.name == 'dense'):
                print(idx,layer.name)
                #print(layer.get_weights())
                print("________________")

pred = f.predict_generator(test_generator, steps = len(test_generator), verbose = 1)

sio.savemat('vggphoto4096.mat',mdict={'feature':pred})

print(pred.shape)









