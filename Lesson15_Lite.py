from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
#import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# table1 print imports
from rich.console import Console
from rich.table import Table

#tf.debugging.set_log_device_placement(True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # load mnist dataset

# change image format
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# normal the images
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255
# Преобразуем ответы в формат one_hot_encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

count_of_n = [10, 100, 5000]
activations = ['relu', 'linear']
bsizes = [1, 10, 100, 60000]

console = Console()
table = Table(show_header=True)
table.add_column('Function of active')
table.add_column('Neouron count')
table.add_column('Batch size')
table.add_column('Accuracy')

models_parameters = []

for f in activations:
    for nc in count_of_n:
        for bs in bsizes:
            # create model
            print(f"Activation: {f}, neuron_count: {nc}, bs: {bs}")
            model = Sequential()
            model.add(Dense(nc, input_dim=784, activation=f))
            model.add(Dense(400, activation=f))
            model.add(Dense(10, activation="softmax"))
            # compile the model
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])

            # train the model
            hist = model.fit(x_train, y_train, batch_size=bs, epochs=15, verbose=1)
            accuracy = hist.history['accuracy'][-1]
            params = [f, nc, bs, accuracy]
            models_parameters.append(params)

for p in models_parameters:
    table.add_row(p[0], str(p[1]), str(p[2]), str(p[3]))

console.print(table)