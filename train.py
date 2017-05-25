import csv
import cv2
import numpy as np
import sklearn
import scipy.ndimage
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samples = []
with open("recorded_data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]

            images = []
            measurements = []

            for batch_sample in batch_samples:
                center = batch_sample[0].split('\\')[-1]
                left = batch_sample[1].split('\\')[-1]
                right = batch_sample[2].split('\\')[-1]
                current_path = "recorded_data/IMG/"
                # image_center = cv2.cvtColor(cv2.imread(current_path + center), cv2.COLOR_BGR2RGB)
                # image_left = cv2.cvtColor(cv2.imread(current_path + left), cv2.COLOR_BGR2RGB)
                # image_right = cv2.cvtColor(cv2.imread(current_path + right), cv2.COLOR_BGR2RGB)
                image_center = scipy.ndimage.imread(current_path + center)
                image_left = scipy.ndimage.imread(current_path + left)
                image_right = scipy.ndimage.imread(current_path + right)
                correction = 1
                center_value = float(batch_sample[3])
                left_value = center_value + correction
                right_value = center_value - correction

                images.extend([image_center, image_left, image_right])
                measurements.extend([center_value, left_value, right_value])

                images.extend([np.fliplr(image_center),np.fliplr(image_left),np.fliplr(image_right)])
                measurements.extend([-1.0 * center_value, right_value, left_value])
                # images.append(image_center)
                # measurements.append(center_value)

                X_train = np.array(images)
                y_train = np.array(measurements)
                yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validator_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, MaxPooling2D, Cropping2D, Dropout

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
# nVidia architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),validation_data=validator_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

# ### print the keys contained in the history object
# print(history_object.history.keys())
#
#
# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save("model2.h5")