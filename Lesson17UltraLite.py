import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from PIL import Image #Для отрисовки изображений

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

#задаём batch_size
batch_size = 128

model = Sequential()
#Первый сверточный слой
model.add(Conv2D(32,(3,3), padding="same", activation='relu', input_shape=(28,28,1)))
#Второй сверточный слой
model.add(Conv2D(32,(3,3), padding="same", activation='relu'))
#Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2,2)))
#Слой регуляризации Dropout
model.add(Dropout(0.25))
#слой выпрямления данных в один вектор
model.add(Flatten())
#Полносвязный слой для классификации
model.add(Dense(256, activation="relu"))
#Слой регуляризации Dropout
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

#Компилируем сеть
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#print NN Arch
model.summary()

#Обучаем сеть на данных mnist
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=15, validation_data=(x_test, y_test), verbose=1)
#plot graffics
loss_train = hist.history["loss"]
loss_val = hist.history["val_loss"]
acc_train = hist.history['accuracy']
acc_val = hist.history["val_accuracy"]

f, (lossPlot, accPlot) = plt.subplots(1, 2)
lossPlot.plot(loss_train, label='Ошибка на обучающем наборе')
lossPlot.plot(loss_val, label='Ошибка на проверочном наборе')
lossPlot.set_xlabel('Эпоха обучения')
lossPlot.set_ylabel('Ошибка')
lossPlot.legend()

accPlot.plot(acc_train, label='Точность на обучающем наборе')
accPlot.plot(acc_val, label='Точность на проверочном наборе')
accPlot.set_xlabel('Эпоха обучения')
accPlot.set_ylabel('Точность')
accPlot.legend()

plt.show()

