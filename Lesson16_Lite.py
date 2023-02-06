from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# table1 print imports
from rich.console import Console
from rich.table import Table

console = Console()
table1 = Table(show_header=True, title="Arch 1, best for previous lessons")
table2 = Table(show_header=True, title="Arch 2")
table3 = Table(show_header=True, title="Arch 3")

table1.add_column("Train_size")
table1.add_column("Test_size")
table1.add_column("Accuracy")

table2.add_column("Train_size")
table2.add_column("Test_size")
table2.add_column("Accuracy")

table3.add_column("Train_size")
table3.add_column("Test_size")
table3.add_column("Accuracy")


#create and train nn with best parameters founded in Lite-level 15 lesson
f = "relu"
nc = 5000
bs = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data() #load mnist dataset

train_sizes = [50000, 10000, 500]
test_sizes = [10000, 50000, 60000-500]

parms = []

parms2 = []

parms3 = []

x_len = x_train.shape[0]
# change image format
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
# normal the images
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255
# Преобразуем ответы в формат one_hot_encoding
y_train = utils.to_categorical(y_train, 10)
# print(y_train)
y_test = utils.to_categorical(y_test, 10)

for i in range(len(train_sizes)):
    x_train_new = x_train[:x_len-train_sizes[i]]
    y_train_new = y_train[:x_len-train_sizes[i]]
    x_test_new = x_train[x_len-train_sizes[i]:]
    y_test_new = y_train[x_len-train_sizes[i]:]

    model1 = Sequential()
    model1.add(Dense(nc, input_dim=784, activation=f))
    model1.add(Dense(400, activation=f))
    model1.add(Dense(10, activation="softmax"))
    # compile the model
    model1.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    hist = model1.fit(x_train_new, y_train_new, batch_size=bs, epochs=15, validation_data=(x_test_new, y_test_new), verbose=1)

    # get best result
    parms += [[train_sizes[i], test_sizes[i], hist.history['accuracy'][-1]]]

    model2 = Sequential()
    model2.add(Dense(1000, input_dim=784, activation=f))
    model2.add(Dense(400, activation=f))
    model2.add(Dense(10, activation="softmax"))
    # compile the model
    model2.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    hist = model2.fit(x_train_new, y_train_new, batch_size=bs, epochs=15, validation_data=(x_test_new, y_test_new),
                      verbose=1)

    # get best result
    parms2 += [[train_sizes[i], test_sizes[i], hist.history['accuracy'][-1]]]

    model3 = Sequential()
    model3.add(Dense(1000, input_dim=784, activation=f))
    model3.add(Dropout(0.5))
    model3.add(BatchNormalization())
    model3.add(Dense(5000, activation=f))
    model3.add(Dropout(0.8))
    model3.add(BatchNormalization())
    model3.add(Dense(400, activation=f))
    model3.add(Dropout(0.6))
    model3.add(BatchNormalization())
    model3.add(Dense(10, activation="softmax"))
    # compile the model
    model3.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    hist = model3.fit(x_train_new, y_train_new, batch_size=bs, epochs=15, validation_data=(x_test_new, y_test_new),
                      verbose=1)

    # get best result
    parms3 += [[train_sizes[i], test_sizes[i], hist.history['accuracy'][-1]]]



for p in parms:
    table1.add_row(str(p[0]), str(p[1]), str(p[2]))

for p in parms2:
    table2.add_row(str(p[0]), str(p[1]), str(p[2]))

for p in parms3:
    table3.add_row(str(p[0]), str(p[1]), str(p[2]))


console.print(table1, table2, table3)