import random

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow import device
from tensorflow.keras.preprocessing import image
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from PIL import Image #Для отрисовки изображений

from rich.console import Console
from rich.table import Table



(x_train, y_train), (x_test, y_test) = mnist.load_data()

console = Console()
table = Table(show_header=True)
table.add_column('Function of active')
table.add_column('Neouron count')
table.add_column('Batch size')
table.add_column('Accuracy')


demo = False

#Выводим для примера картинки по каждому классу
if demo:
    fig, axs = plt.subplots(1, 10, figsize=(25, 3)) #Создаем полотно из 10 графиков
    for i in range(10): #Проходим по классам от 0 до 9
      label_indexes = np.where(y_train==i)[0] #Получаем список из индексов положений класса i в y_train
      index = random.choice(label_indexes) #Случайным образом выбираем из списка индекс
      img = x_train[index] #Выбираем из x_train нужное изображение
      axs[i].imshow(Image.fromarray(img), cmap='gray') #Отображаем изображение i-ым графиков

    plt.show() #Показываем изображения

# y_test, y_train to OHE
y_test = utils.to_categorical(y_test, 10)
y_train = utils.to_categorical(y_train, 10)

#Меняем формат данных MNIST
#Надо добавить в конце размерность 1
#Чтобы свёрточная сеть понимала, что это чёрно-белые данные
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Посмотрим форматы выборок перед обучением
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


neuron_count = [2, 4, 16]
activations = ["relu", "linear"]
batch_sizes = [10, 100, 10000]

params = []
with device("/CPU:0"):
    for f in activations:
        for nc in neuron_count:
            for bs in batch_sizes:
                print(F"Train ach with: activation: {f}, neuron count {nc}, batch size: {bs}")
                #time.sleep(2)
                model = Sequential()
                # Первый сверточный слой
                model.add(Conv2D(nc, (3, 3), padding="same", activation=f, input_shape=(28, 28, 1)))
                # Второй сверточный слой
                model.add(Conv2D(32, (3, 3), padding="same", activation=f))
                # Первый слой подвыборки
                model.add(MaxPooling2D(pool_size=(2, 2)))
                # Слой регуляризации Dropout
                model.add(Dropout(0.25))
                # слой выпрямления данных в один вектор
                model.add(Flatten())
                # Полносвязный слой для классификации
                model.add(Dense(256, activation="relu"))
                # Слой регуляризации Dropout
                model.add(Dropout(0.25))
                model.add(Dense(10, activation="softmax"))

                # Компилируем сеть
                model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
                # Обучаем сеть на данных mnist
                hist = model.fit(x_train, y_train, batch_size=bs, epochs=15, validation_data=(x_test, y_test),
                                 verbose=1)
                accuracy = hist.history["accuracy"][-1]
                params += [[f, str(nc), str(bs), str(accuracy)]]
                del model
for p in params:
    table.add_row(p[0], p[1], p[2], p[3])
console.print(table)
