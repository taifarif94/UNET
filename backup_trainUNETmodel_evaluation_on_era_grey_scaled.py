# Code taken from:
# Reference: martijnfolmer. 2023. Super Resolution using UNET. https://github.com/
# martijnfolmer/SuperResolution_using_UNET GitHub repository.
# Slight modifications

# USER VARIABLES
pathToFolderWithImages = '/content/output_images_grey_from_npy'  # absolute path to the folder containing the images of your dataset
pathToNewTestDataset = '/content/SuperResolution_using_UNET_for_temperature_downscaling/output_images_grey_from_grib'  # Path to the new dataset for evaluation

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from CreateUNETmodel import CreateUNET

img_size_small = (64, 64)  # The size of small images to scale up from
img_size_big = (256, 256)  # The size of big images we want to output
offset = 0.0  # The offset and scale to preprocess the images
scale = 1.0

batch_size = 8  # The batch size during training
num_epochs = 1  # Number of epochs we train

load_model = False  # If set to true, we load a previous model to train on
load_model_path = ''  # The model we load, in case load_model = True

pathToResultingImg = 'test_evaluations'  # Where we save the results of our evaluation
pathToResultingModel = 'resulting_model'  # Where we save our model when finished training
tflite_name = 'UNET.tflite'  # The name of the saved tflite model

# Clear previous Keras session
keras.backend.clear_session()

class DataGenerator(keras.utils.Sequence):
    def __init__(self, all_filenames, batch_size=8, img_size_small=(64, 64), img_size_big=(256, 256),
                 offset=0.0, scale=1.0 / 255.0, shuffle_index=True):
        self.X_filenames = all_filenames
        self.batch_size = batch_size
        self.offset = offset
        self.scale = scale
        self.indices = np.arange(self.X_filenames.shape[0])
        self.shuffle_index = shuffle_index
        self.imgSize_small = img_size_small
        self.imgSize_big = img_size_big

    def LoadImages(self, batchxFilenames):
        allImg = [cv2.imread(filename) for filename in batchxFilenames]
        return allImg

    def randomlyFlip(self, allImg):
        for i_img, img_c in enumerate(allImg):
            if random.random() < 0.5:
                allImg[i_img] = cv2.flip(img_c, 1)
        return allImg

    def setOffsetAndScale(self, allImg):
        allImg = [(img[:, :, :] + self.offset) * self.scale for img in allImg]
        return allImg

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X_filenames = self.X_filenames[inds]
        allImg = self.LoadImages(batch_X_filenames)
        allImg = self.randomlyFlip(allImg)
        allBigImg = [cv2.resize(img, self.imgSize_big) for img in allImg]
        allSmallImg = [cv2.resize(img, self.imgSize_small) for img in allBigImg]
        allSmallImg = [cv2.resize(img, self.imgSize_big) for img in allSmallImg]
        allBigImg = self.setOffsetAndScale(allBigImg)
        allSmallImg = self.setOffsetAndScale(allSmallImg)
        return np.asarray(allSmallImg), np.asarray(allBigImg)

    def __len__(self):
        return int(np.ceil(len(self.X_filenames) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle_index:
            np.random.shuffle(self.indices)

# Load or build the model
if load_model and os.path.exists(load_model_path):
    model = keras.models.load_model(load_model_path)
else:
    CU = CreateUNET(input_shape=(img_size_big[0], img_size_big[1], 3))
    model = CU.get_model()
model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['acc'])

# Prepare the original dataset
X_filenames = np.asarray([os.path.join(pathToFolderWithImages, f) for f in os.listdir(pathToFolderWithImages) if f.endswith('.png')])
X_train, _ = train_test_split(X_filenames, test_size=0.2, random_state=76)

# Train the model
train_gen = DataGenerator(X_train, batch_size=batch_size, img_size_small=img_size_small, img_size_big=img_size_big, offset=offset, scale=scale)
model.fit(train_gen, epochs=num_epochs, verbose=1)

# Save the trained model
if not os.path.exists(pathToResultingModel):
    os.makedirs(pathToResultingModel)
model.save(pathToResultingModel)

# Prepare the new test dataset
X_new_test = np.asarray([os.path.join(pathToNewTestDataset, f) for f in os.listdir(pathToNewTestDataset) if f.endswith('.png')])

# Ensure the directory for test evaluations exists
if not os.path.exists(pathToResultingImg):
    os.makedirs(pathToResultingImg)

# Evaluate model using the new dataset and save every 5th image
for i, filename in enumerate(X_new_test):
    if i % 5 == 0:  # Process every 5th image
        img = cv2.imread(filename)
        low_res_img = cv2.resize(img, img_size_small)
        low_res_img = cv2.resize(low_res_img, img_size_big)
        model_input = (low_res_img.astype(np.float32) + offset) * scale
        model_input = np.expand_dims(model_input, axis=0)
        high_res_output = model.predict(model_input)[0]
        high_res_output = (high_res_output / scale) - offset
        high_res_output = np.clip(high_res_output, 0, 255).astype('uint8')

        # Convert both images to grayscale
        low_res_gray = cv2.cvtColor(low_res_img, cv2.COLOR_RGB2GRAY)
        high_res_gray = cv2.cvtColor(high_res_output, cv2.COLOR_RGB2GRAY)

        # Scale the evaluated grayscale image to match the input image range
        min_input, max_input = np.min(low_res_gray), np.max(low_res_gray)
        min_output, max_output = np.min(high_res_gray), np.max(high_res_gray)
        high_res_gray_scaled = (high_res_gray - min_output) / (max_output - min_output) * (max_input - min_input) + min_input
        high_res_gray_scaled = np.clip(high_res_gray_scaled, 0, 255).astype(np.uint8)

        # Convert back to RGB for stacking
        high_res_gray_rgb = cv2.cvtColor(high_res_gray_scaled, cv2.COLOR_GRAY2RGB)
        low_res_gray_rgb = cv2.cvtColor(low_res_gray, cv2.COLOR_GRAY2RGB)

        # Combine low resolution and high resolution images
        combined_image = np.hstack((low_res_gray_rgb, high_res_gray_rgb))
        cv2.imwrite(os.path.join(pathToResultingImg, f'eval_image_{i}.png'), combined_image)

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(tflite_name, 'wb') as f:
    f.write(tflite_model)
