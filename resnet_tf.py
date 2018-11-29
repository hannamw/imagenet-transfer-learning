import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.utils.np_utils import to_categorical
import sklearn
import cv2
import pathlib
import time

train_dir = pathlib.Path("data")

categories = {"spider":0, "scorpion": 1}


data = {"train":{"images": [], "labels": []}, "val":{"images": [], "labels": []},"test":{"images":[], "labels":[]}}


for subdir in train_dir.iterdir():
    folder = subdir.name
    for image_category in subdir.iterdir():
        clabel = categories[image_category.name]
        for image in image_category.iterdir():
            imr = cv2.imread(str(image), 1)
            if not(imr is None):
                data[folder]["images"].append(cv2.resize(imr, (224, 224)))
                data[folder]["labels"].append(clabel)

for folder_type in data.keys():
    data[folder_type]["images"] = np.array([np.array(image, dtype = "float") for image in data[folder_type]["images"]], dtype = "float")/255.0
    data[folder_type]["labels"] = to_categorical(np.array(data[folder_type]["labels"]), num_classes = 2)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.end_time = time.time()
        self.times.append(self.end_time-self.start_time)
timer = TimeHistory()

resnet = keras.applications.resnet50.ResNet50(input_shape = (224,224,3), include_top=False, pooling="avg", weights="imagenet")


model = Sequential()
model.add(resnet)
model.add(Dense(2, activation = "softmax"))

print(model.layers[:-1][0].layers)
for layer in model.layers[:-1][0].layers:
  layer.trainable = False
  #due to default batch normalization behavior, we need to take batch normalization
  #layers out of training mode, otherwise they will use the wrong weights
  #causing mono-class prediction
  if(isinstance(layer,BatchNormalization)):
     layer.training = False
  
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(data["train"]["images"], data["train"]["labels"], epochs = 25, validation_data = (data["val"]["images"], data["val"]["labels"]), batch_size = 20, callbacks = [timer])

print(timer.times)
print(sum(timer.times[1:])/(len(timer.times)-1))

start_time = time.clock()
results = model.evaluate(data["test"]["images"], data["test"]["labels"])
end_time = time.clock()
print(model.metrics_names)
print(results)
print("The time taken to evaluate was %f" % (end_time-start_time))
