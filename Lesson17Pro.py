from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow import device
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np #Библиотека работы с массивами
import matplotlib.pyplot as plt #Для отрисовки графиков
from PIL import Image #Для отрисовки изображений

from rich.console import Console
from rich.table import Table


def research(x_train, y_train, x_test, y_test,  count_of_conv=1, count_of_neurons=32, coef_of_dropout=0.25, bath_size=100, ismaxpooling=True, isdropout=True, isbatch=True):
    '''
            Функция research создает сверточную сеть.
                Входные параметры:
                count_conv    - количество сверточных слоев;
                count_neurons - количество нейронов в сверточном слое;
                ismaxpooling  - добавляем или нет макспулинги;
                isdropout     - добавляем или нет дропауты;
                coef_of_dropout -- коефициент дропаута
                isbatch       - добавляем или нет батч нормализацию.
                batch_size - размер батча

        '''
    x_train = np.concatenate([x_train, x_test], axis=0)
    y_train = np.concatenate([y_train, y_test], axis=0)
    print(x_train.shape)
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # Значения цвета меняем на дробные показания
        rotation_range=10,  # Поворачиваем изображения при генерации выборки
        width_shift_range=0.1,  # Двигаем изображения по ширине при генерации выборки
        height_shift_range=0.1,  # Двигаем изображения по высоте при генерации выборки
        zoom_range=0.1,  # Зумируем изображения при генерации выборки
        horizontal_flip=True,  # Включаем отзеркаливание изображений
        fill_mode='nearest',  # Заполнение пикселей вне границ ввода
        validation_split=0.2  # Указываем разделение изображений на обучающую и тестовую выборку
    )
    datagen.fit(x_train)
    train_gen =datagen.flow(x_train, y_train, batch_size=bath_size,
         subset='training')
    validation_gen = datagen.flow(x_train, y_train,
         batch_size=bath_size, subset='validation')

    model = Sequential()
    model.add(BatchNormalization(input_shape=(32,32,3)))
    for f in range(count_of_conv):
        model.add(Conv2D(count_of_neurons, (3, 3), padding='same', activation='relu'))
        if (ismaxpooling == True):
            model.add(MaxPooling2D(pool_size=(2, 2)))

        if isdropout:
            model.add(Dropout(coef_of_dropout))

        if (isbatch):
            model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if (isdropout == True):
        model.add(Dropout(0.25))
    model.add(Dense(100, activation='softmax'))
    # Компилируем сеть
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    hist = model.fit_generator(train_gen,
                               epochs=40,
                               validation_data= validation_gen,
                               verbose=1)
    accuracy = hist.history["val_accuracy"][-1]
    return accuracy


(x_train, y_train), (x_test, y_test) = cifar100.load_data()
count_of_n = [128, 256, 400]
print(x_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_train.shape)

y_test = utils.to_categorical(y_test, 100)
y_train = utils.to_categorical(y_train, 100)

console = Console()
table = Table(show_header=True)
table.add_column("Count of conv layers")
table.add_column("Count of neourons in one conv")
table.add_column("Accuracy on test data")

for coc in range(2,6):
    for con in count_of_n:
        print(f"Counf_of_conv:{coc} count_of_N:{con}")
        try:
            accuracy = research(x_train, y_train, x_test, y_test, count_of_conv=coc, count_of_neurons=con, bath_size=6)
        except Exception as e:
            print(e)
            accuracy = 0

        table.add_row(str(coc), str(con), str(accuracy))

console.print(table)