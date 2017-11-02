# Import all libraries that are needed
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt

# Start the process getting the initial time
start_time = time.time()
print("Start process")

# -------------------------------------------------- Organizing Data ---------------------------------------------------
# Size of batch to get the images in parts using generator instead of getting all at the same time in the memory
batch_size = 340  # 340 was a good number for me, higher values means fast training, however it will use more memory


# Generator to provide the images as they are used, avoiding to load all of them in the memory in the same time
def generator(samples, batch_size_in=256, correction=0.2,
              base_path_image='/media/marlosdamasceno/Dados/Udacity/Self Drive Car/Simulator/DataTrack1/IMG/'):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)  # Randomly mix the images
        for offset in range(0, num_samples, batch_size_in):  # Get the batches from all images
            batch_samples = samples[offset:offset + batch_size_in]
            images_in = []  # It will store the images
            angles = []  # It will store the angles
            for batch_sample in batch_samples:
                steering_center = float(batch_sample[3])  # Get the center steering angle
                name = base_path_image + batch_sample[0].split('/')[-1]  # Get the center image
                center_image = cv2.imread(name)
                images_in.append(center_image)
                angles.append(steering_center)

                # create adjusted steering measurement for the LEFT side camera image
                steering_left = steering_center + correction  # Get the LEFT steering angle
                name = base_path_image + batch_sample[1].split('/')[-1]  # Get the LEFT image
                left_image = cv2.imread(name)
                images_in.append(left_image)
                angles.append(steering_left)

                # create adjusted steering measurement for the RIGHT side camera image
                steering_right = steering_center - correction  # Get the RIGHT steering angle
                name = base_path_image + batch_sample[2].split('/')[-1]  # Get the RIGHT image
                right_image = cv2.imread(name)
                images_in.append(right_image)
                angles.append(steering_right)

            # Store the images and angles as numpy arrays
            X_train = np.array(images_in)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


lines = []  # Store the lines from the log file
# Repository where the log file is. Chenge to the loction of your file
base_path = '/media/marlosdamasceno/Dados/Udacity/Self Drive Car/Simulator/DataTrack1/'
with open(base_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)  # Split the data into train and validation

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size_in=batch_size)
validation_generator = generator(validation_samples, batch_size_in=batch_size)

# ----------------------------------------------- Architecture of the model --------------------------------------------
# ----------------------------------------------- Nvidia + Dropout -----------------------------------------------------
model = Sequential()
# Change the mean to zero and the image to float type
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# Cropping the image 70 on top and 25 on bottom
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# Five convolution layers
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 3, 3, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
# Flatten the layer
model.add(Flatten())
# A fully connected layer
model.add(Dense(100, activation='relu'))
# Flowed by a dropout to avoid over-fitting
model.add(Dropout(rate=0.5))
# A fully connected layer
model.add(Dense(50, activation='relu'))
# Flowed by a dropout to avoid over-fitting
model.add(Dropout(rate=0.5))
# Finishing with to fully connected layers
model.add(Dense(10))
model.add(Dense(1))

# ------------------------------------------------- Training the model -------------------------------------------------
model.compile(loss='mse', optimizer='adam')  # Using mse and adam
# Store the data (training and validation loss) of each epoch (verbose = 1). Use the generators to train
# Runs in 30 epochs
history_object = model.fit_generator(train_generator, steps_per_epoch=int(len(train_samples) / batch_size + 1),
                                     validation_data=validation_generator,
                                     nb_val_samples=int(len(validation_samples) / batch_size + 1), epochs=30, verbose=1)
# -------------------------------------------------- Saving the model --------------------------------------------------
model.save('model.h5')

# Print the total time spent to training
print("--- Total time: %s seconds ---" % (time.time() - start_time))

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
