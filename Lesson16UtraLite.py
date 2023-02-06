from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
# sklearn - популярная библиотека для машинного обучения
# train_test_split - функция разделения на обучающую и проверочную/тестовую выборку
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# table1 print imports
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(show_header=True)

table.add_column("test_size")
table.add_column("Accuracy")


(x_train_org, y_train_org), (x_test, y_test) = mnist.load_data() #load mnist dataset

test_sizes = [0.1, 0.5, 0.9]

parms = []

for ts in test_sizes:
    x_train, x_test, y_train, y_test = train_test_split(x_train_org, y_train_org, test_size=ts, shuffle=True)

   # print(x_train.shape)
    #change image format
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    # normal the images
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255
    # Преобразуем ответы в формат one_hot_encoding
    y_train = utils.to_categorical(y_train, 10)
    #print(y_train)
    y_test = utils.to_categorical(y_test, 10)

    #create model
    model = Sequential(name ='first_ai')
    model.add(Dense(800, input_dim=784, activation="relu"))
    model.add(Dense(400, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    #compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    #print(model.summary())

    #train the model
    hist = model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)
    #get best result
    parms += [[ts, hist.history['accuracy'][-1]]]

#print(parms)
for p in parms:
    table.add_row(str(p[0]), str(p[1]))

console.print(table)