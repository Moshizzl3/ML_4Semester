import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import load_img, img_to_array


# Set constants
targetSize = 200  # pixel dimension after ImageDataGenerator has processed
no_of_filters = 10  # how many different filters
color = 'rgb'  # use "rgb" for color images
# use 'categorical' for >2 class, 'binary' for two-class problems
classMode = 'categorical'
trainingFiles = 'week_40/cat-dog/training_set12'  # change according to your setup


# Image augmentation.
# Prepare images before training the model.
train_datagen = ImageDataGenerator(
    rescale=1./255,  # change pixel value from 0-255 to 0.0 - 1.0
    shear_range=0.2,  # distort the image sideways
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range= 0.1,
    horizontal_flip=True
)
# you can also use shear_range, zoom_range and horizontal_flip to "disturb" the images.
# This will make the model more robust, and will reduce overfitting.


# Optional: Also create a test generator, to generate a test_set in next step.

# Optional:
# no zoom or shear, since this is test data
test_datagen = ImageDataGenerator(rescale=1./255)


# Create a training set object. This will be given as argument to model.fit(...) later.
training_set = train_datagen.flow_from_directory(
    trainingFiles,  # path to folder with images
    # size of output image, f.x. 28 x 28 pixel
    target_size=(targetSize, targetSize),
    batch_size=32,  # how many images to load at a time
    class_mode=classMode,  # use 'categorical' for >2 class, 'binary' for two-class problems
    color_mode=color)   # use 'grayscale' for black/white, 'rgb' for color


# Optional:
# Create a test set object: this is done exactly like the training set, except for the path
# Optional:
test_set = test_datagen.flow_from_directory(
    'week_40/cat-dog/test_set12',  # path to folder with test images
    # size of output image, f.x. 28 x 28 pixel
    target_size=(targetSize, targetSize),
    batch_size=32,  # how many images to load at a time
    class_mode=classMode,  # use 'categorical' for >2 class, 'binary' for two-class problems
    color_mode=color)   # use 'grayscale' for black/white, 'rgb' for color

# Create model.
# Here we add the first convolutional layer.
# It will run filters across the input image, and output a new, smaller image. Where pixel values that match the filters are preserved.

model = Sequential()  # instantiate new model object.

model.add(Conv2D(filters=32,  # specify number of filters. Higher number for more complex images.
                 kernel_size=3,  # size of filter - typically 3, as in 3x3
                 activation="relu",  # activation function, often 'relu' on layers before last
                 kernel_initializer="he_uniform",
                 padding="same",
                 input_shape=[targetSize, targetSize, 3]))  # dimension of image, coming in from training set.
# here 28x28 pixel. '1' is number of color channels. B/W = 1, Color = 3.
# Max Pooling.
# Scan 4 pixels, capture the one with highest pixel value. Then move 2 pixels to the right and repeat.
# reduce the image size. Here 4 pixels will become 1 pixel.
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.1))
# pool_size is the size of the square which will be converted to just one pixel. Often 2, as in 2x2.
# strides is how many pixels to move to the right after each pooling operation. Often 2.
# You are free to add more layers of Conv2D and MaxPool2D to make the model smarter

model.add(Conv2D(64, kernel_size=3, activation="relu", kernel_initializer="he_uniform",
          padding="same", input_shape=[targetSize, targetSize, 1]))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=3, activation="relu", kernel_initializer="he_uniform",
          padding="same",
          input_shape=[targetSize, targetSize, 1]))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(0.2))


# Flatten.
# Convert a 2d array to a 1d array

# Convert the output layer to a single column, an array of shape (length, 1).
model.add(Flatten())

# Add hidden layer(s) and output layer, just as in DNN Step 11 above.
# add fully connected layer (just as with DNN Step 11 above)
# here 4 output neurons for a simple problem
model.add(Dense(units=64, activation="relu", kernel_initializer="he_uniform"))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# one single output neuron and sigmoid ac. func.
model.add(Dense(units=2, activation="softmax"))

# Call model.compile(..) which will prepare the model to be trained later.

# use the Adam optimizer, and specify learning-rate
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])
# here parameters are set to solve a two-class problem. Hence binary_crossentropy.
# metrics = ['accuracy'] should be used for classification (not regression)

# Train
# This will train the model
model.fit(x=training_set, epochs=40, validation_data=test_set)
# if you made a test_set in step 5, then provide it as parameter like this: validation_data=test_set
# The model will evaluate against the test set at the end of each epoch. The model will not be trained on these images.

# Figure out, what number (0 or 1) each class belongs to.
# get the indices for each class. F.x. horizontal = 0, vertical = 1
print(training_set.class_indices)

# Optional:
# Get model prediction accuracy on test data

# If you made a test_set in step 5, this will predict ALL the images and return the accuracy in %.
model.evaluate(test_set)
# the reason it can be evaluated is, that the labels are automatically inferred from the folder structure of the image files.


# Predict a single new value. Here we need to prepare the image first
singlePred = 'week_40/cat-dog/single_prediction/cat_or_dog_1.jpg'
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

model.save("./model_catordog.h5")
