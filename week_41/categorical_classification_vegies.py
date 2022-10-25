import numpy as np
import tensorflow as tf
import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import load_img, img_to_array

# Set constants
targetSize = 62
batchSize = 32
classMode = 'categorical'
color = 'rgb'
trainingFiles = 'week_41/vegies/train'
testFiles = 'week_41/vegies/test'
validationFiles = 'week_41/vegies/validation'


# Image augmentation.
# Prepare images before training the model.
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    trainingFiles, target_size=(targetSize, targetSize), batch_size=batchSize, class_mode=classMode, color_mode=color)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    testFiles, target_size=(targetSize, targetSize), batch_size=batchSize, class_mode=classMode, color_mode=color)


validation_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)
validation_set = val_datagen.flow_from_directory(
    testFiles, target_size=(targetSize, targetSize), batch_size=batchSize, class_mode=classMode, color_mode=color)

# Create model.
# Here we add the first convolutional layer.
# It will run filters across the input image, and output a new, smaller image. Where pixel values that match the filters are preserved.

model = Sequential()  # instantiate new model object.

model.add(Conv2D(filters=64, kernel_size=3, activation="relu",
          input_shape=[targetSize, targetSize, 3]))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, activation="relu",
          input_shape=[targetSize, targetSize, 3]))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Conv2D(filters=32, kernel_size=3, activation="relu",
          input_shape=[targetSize, targetSize, 3]))
model.add(MaxPool2D(pool_size=2, strides=2))



# Flatten.
# Convert a 2d array to a 1d array
# Convert the output layer to a single column, an array of shape (length, 1).
model.add(Flatten())

# Add hidden layer(s) and output layer
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=32, activation="relu"))
model.add(Dense(units=15, activation="softmax"))

# Call model.compile(..) which will prepare the model to be trained later.
# here parameters are set to solve a multi-class problem. Hence categoricail_crossentropy.
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.optimizer.get_config())
# create logfiles for tensorboard
log_dir = "./week_41/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Train
# This will train the model
model.fit(x=training_set, epochs=50, validation_data=validation_set,
          callbacks=[tensorboard_callback])

# Figure out, what number (0 or 1) each class belongs to.
# get the indices for each class. F.x. horizontal = 0, vertical = 1
print(training_set.class_indices)

# the reason it can be evaluated is, that the labels are automatically inferred from the folder structure of the image files.
model.evaluate(test_set)


# Predict a single new value. Here we need to prepare the image first
singlePred = 'week_40/myTests/2-2.png'
test_image = image.image_utils.load_img(
    singlePred, target_size=[targetSize, targetSize], color_mode=color)
# here set the SAME parameters as on the training in Step 5.
# possible solution: image.image_utils.load_img  Set image_utils after image.

test_image = image.image_utils.img_to_array(
    test_image)  # convert image to array
# add one extra dimension to hold batch.
test_image = np.expand_dims(test_image, axis=0)
# axis=0 means that a new dimension will be added, such that test_image.shape goes from (28, 28, 1) to (1, 28, 28, 1). This is required by Tensorflow.

# remember to divide each pixel value by 255.0
result = model.predict(test_image/255.0)
result = np.array(result)


print(result)
print(np.argmax(result))


model.save("./model_vegies1.h5")
