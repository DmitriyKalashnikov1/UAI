from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # НС прямого распространения
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization # Основные слои
from tensorflow.keras import utils # Утилиты для to_categorical
from tensorflow.keras.preprocessing import image # Для отрисовки изображения
from sklearn import preprocessing
from tensorflow.keras.optimizers import Adam, Adadelta # Алгоритмы оптимизации, для настройки скорости обучения
import numpy as np # Библиотека работы с массивами
import matplotlib.pyplot as plt # Отрисовка изображений
from PIL import Image # Отрисовка изображений
import pandas as pd # Библиотека pandas


test_size = 1000

def to_dict(s):
    ret = {}  # Создаём пустой словарь
    for _id, name in enumerate(s):  # Проходим по всем парам - id и название
        ret.update({name: _id})  # Добавляем в словарь
    return ret

def to_ohe(value, d):
    arr = [0] * len(d)
    arr[d[value]] = 1
    return arr

inCollab = False

if inCollab:
    from google.colab import drive
    drive.mount('/content/drive')
    path = "/content/drive/MyDrive/resouces/"
else:
    path = "./resouces/"

cars = pd.read_csv(path + "cars_new.csv", sep=",")

print(cars.head())

print(cars.values.shape)

#выделяем и сохраняем отдельно текстовые коллнки датасета
marks_dict = to_dict(set(cars["mark"]))
models_dict = to_dict(set(cars["model"]))
bodies_dict = to_dict(set(cars['body']))
kpps_dict = to_dict(set(cars['kpp']))
fuels_dict = to_dict(set(cars['fuel']))

# Запоминаем цены
prices = np.array(cars['price'], dtype=np.float)

# Запоминаем числовые параметры
# и нормируем
years = preprocessing.scale(cars['year'])
mileages = preprocessing.scale(cars['mileage'])
volumes = preprocessing.scale(cars['volume'])
powers = preprocessing.scale(cars['power'])

print(years)
print(marks_dict)

# Создаём пустую обучающую выборку
x_train = []
y_train = []

print("Preparing train and test datesets....")
# Проходим по всем машинам
for _id, car in enumerate(np.array(cars)):
    # В y_train добавляем цену
    y_train.append(prices[_id])

    # В x_train объединяем все параметры
    # Категорийные параметры добавляем в виде ohe
    # Числовые параметры добавляем напрямую
    x_tr = to_ohe(car[0], marks_dict) + \
           to_ohe(car[1], models_dict) + \
           to_ohe(car[5], bodies_dict) + \
           to_ohe(car[6], kpps_dict) + \
           to_ohe(car[7], fuels_dict) + \
           [years[_id]] + \
           [mileages[_id]] + \
           [volumes[_id]] + \
           [powers[_id]]

    # Добавляем текущую строку в общий x_train
    x_train.append(x_tr)

# Превращаем лист в numpy.array
x_train = np.array(x_train, dtype=np.float)
y_train = np.array(y_train, dtype=np.float)

print(x_train.shape, y_train.shape)

# Нормализуем y_train
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()

# y_train.reshape(-1,1) добавляет одну размерность
# Это нужно потому, что y_scaler.fit_transform
# Требует двумерны вектор, массив примеров, которые надо нормализовать
# Он не умеет работать с одним примеров
# Поэтому мы делаем массив из одного примера
# На выходе он так же выдаёт массив примеров
# Но нам нужен только первый пример
# Поэтому мы делаем flatten() - уменьшение размерности


x_test = x_train[-test_size:]
y_test = y_train_scaled[-test_size:]

train_len = len(x_train)

x_train = x_train[0:train_len-test_size]
y_train_scaled = y_train_scaled[0:train_len-test_size]

print("Done!")
print(f"x_train shape:{x_train.shape}, x_test shape:{x_test.shape}, y_train shape:{y_train_scaled.shape}, y_test shape:{y_test.shape}")
#input()

model_ula = Sequential()
model_ula.add(Dense(300, activation = 'relu', input_shape = x_train.shape[1:]))
model_ula.add(Dropout(0.8))
model_ula.add(Dense(100, activation = 'relu'))
model_ula.add(Dropout(0.8))
model_ula.add(Dense(1))
model_ula.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
# training
hist = model_ula.fit(x_train, y_train_scaled, epochs = 20, batch_size = 32, validation_split = 0.2)

predict = model_ula.predict(x_test)
#print(predict)

predict_inverse = y_scaler.inverse_transform(predict.reshape(-1,1)).flatten()
y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

#считаем среднюю ошибку, процент ошибок и среднюю цену
delta = y_test_unscaled - predict_inverse
abs_delta = abs(delta)
mean_delta = sum(abs_delta)/len(abs_delta)
mean_price = sum(y_test_unscaled)/len(y_test_unscaled)

print(f"Mean error: {mean_delta}, Average price:{mean_price}, Average error percent {round(100*mean_delta / mean_price)}%")


print(hist.history.keys())
#plot graffics
loss_train = hist.history["loss"]
loss_val = hist.history["val_loss"]
acc_train = hist.history['mae']
acc_val = hist.history["val_mae"]

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
