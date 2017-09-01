import keras

#Part 1: Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import  Dense

classifier = Sequential()

#Step 1 : Convolution
#Convolution step uses the stride of 1
#creating 32 feature detectors of 3*3 dimension whch means our CNN will have 32 feature maps.We can increase the size to 64,128,...
#We can add more more convolution layers containing 64,128 feature maps to improve the results.
#input shape define the format of the image. Colored images(RGB = 3) of 256 * 256 dim.But we will use small format
# In tensorflow backend , the order is 2D dim and input channel
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation='relu'))

#Step2: pooling
#pooling step uses stride of 2.
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Adding the second Convolution layer to improve the accuracy.
#Since the input shape is pooled featured maps from previous step, we don't need to specify them.
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


##We can add many convolution layers with increased number of feature detectors.


#Step3: Flattening
classifier.add(Flatten())

#Step 4 : Full connection
#Flatten step previously will act as an input layer.
classifier.add(Dense(output_dim=128,activation='relu')) #1st hidden layer
classifier.add(Dense(output_dim=1,activation='sigmoid')) #output layer

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Part2: Fitting the CNN to images.

#performating data augmentation. This to to enrich our dataset to get more diverse images by transforming, rotating and other.
#From the small amount of datasset, we take batches of images geneate multiple images. This helps to avoid overfitting.

#This code wiill preprocess the images, apply data augmentation and fit CNN.
from keras.preprocessing.image import  ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#target size should be same as input_shape parameter choosen earlier.
training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
#steps_per_epoch and validation_steps are training and test sizes we have in our dataset.
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, 
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

