
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow import device
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from PIL import Image #Для отрисовки изображений
from pathlib import Path

baseP = Path('.') / 'resouces' / 'Lesson17AutoBase'

#print(baseP)

batch_size = 15 #Размер выборки
img_width = 96 #Ширина изображения
img_height = 54 #Высота изображения

acc = []
val_acc = []

#Генератор изображений
datagen = ImageDataGenerator(
    rescale=1. / 255, #Значения цвета меняем на дробные показания
    rotation_range=10, #Поворачиваем изображения при генерации выборки
    width_shift_range=0.2, #Двигаем изображения по ширине при генерации выборки
    height_shift_range=0.2, #Двигаем изображения по высоте при генерации выборки
    zoom_range=0.2, #Зумируем изображения при генерации выборки
    horizontal_flip=True, #Включаем отзеркаливание изображений
    fill_mode='nearest', #Заполнение пикселей вне границ ввода
    validation_split=0.2 #Указываем разделение изображений на обучающую и тестовую выборку
)


# обучающая выборка
train_generator = datagen.flow_from_directory(
    baseP / 'train', #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто 
    shuffle=True, #Перемешивание выборки
    subset='training' # устанавливаем как набор для обучения
)

# проверочная выборка
validation_generator = datagen.flow_from_directory(
    baseP / 'val', #Путь ко всей выборке выборке
    target_size=(img_width, img_height), #Размер изображений
    batch_size=batch_size, #Размер batch_size
    class_mode='categorical', #Категориальный тип выборки. Разбиение выборки по маркам авто 
    shuffle=True, #Перемешивание выборки
    subset='validation' # устанавливаем как валидационный набор
)

# Создаем последовательную модель
model = Sequential()
model.add(BatchNormalization())
# Первый сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=(img_width, img_height, 3)))
# Второй сверточный слой
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Третий сверточный слой
#model.add(Dropout(0.35))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# Второй сверточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Третий сверточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.3))
# Четвертый сверточный слой
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации Dropout
model.add(Dropout(0.3))
# Пятый сверточный слой
#model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
# Шестой сверточный слой
#model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(3, 3)))
# Слой регуляризации Dropout
model.add(Dropout(0.3))
# Слой преобразования двумерных данных в одномерные 
model.add(Flatten())
# Полносвязный слой
model.add(Dense(2048, activation='elu'))
# Полносвязный слой
model.add(Dense(4096, activation='elu'))
# Вызодной полносвязный слой
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

#model.summary()
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator, 
            epochs=50,
            verbose=1)

acc += history.history['accuracy']
val_acc += history.history['val_accuracy']
   #Два этапа дообучения с уменшением lr
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.000001), metrics=['accuracy'])
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator, 
            epochs=25,
            verbose=1
        )
acc += history.history['accuracy']
val_acc += history.history['val_accuracy']
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0000001), metrics=['accuracy'])
history = model.fit_generator(
            train_generator,
            validation_data = validation_generator, 
            epochs=20,
            verbose=1
        )
acc += history.history['accuracy']
val_acc += history.history['val_accuracy']

#Оображаем график точности обучения
plt.plot(acc, 
         label='Доля верных ответов на обучающем наборе')
plt.plot(val_acc, 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()