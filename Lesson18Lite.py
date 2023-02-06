import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential # Полносвязная модель
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation # Слои для сети
from tensorflow.keras.preprocessing.text import Tokenizer # Методы для работы с текстами и преобразования их в последовательности
from tensorflow.keras.preprocessing.sequence import pad_sequences # Метод для работы с последовательностями

from sklearn.preprocessing import LabelEncoder # Метод кодирования тестовых лейблов
from sklearn.model_selection import train_test_split # Для разделения выборки на тестовую и обучающую

from tensorflow import device
#from tensorflow.python.framework.errors import ResourceExhaustedError
from pathlib import Path
import time

baseP = Path('.') / 'resouces' / 'Lesson18Writers'


def readText(filename: Path):
    text = ''
    with open(filename, 'r') as readFile:
        text = readFile.read()
        text = text.replace("\n", " ")
    return text


# Формирование обучающей выборки по листу индексов слов
# (разделение на короткие векторы)
def getSetFromIndexes(wordIndexes, xLen, step):  # функция принимает последовательность индексов, размер окна, шаг окна
    xSample = []  # Объявляем переменную для векторов
    wordsLen = len(wordIndexes)  # Считаем количество слов
    index = 0  # Задаем начальный индекс

    while (index + xLen <= wordsLen):  # Идём по всей длине вектора индексов
        xSample.append(wordIndexes[index:index + xLen])  # "Откусываем" векторы длины xLen
        index += step  # Смещаеммся вперёд на step

    return xSample


# Формирование обучающей и проверочной выборки
# Из двух листов индексов от двух классов
def createSetsMultiClasses(wordIndexes, xLen,
                           step):  # Функция принимает последовательность индексов, размер окна, шаг окна

    # Для каждого из 6 классов
    # Создаём обучающую/проверочную выборку из индексов
    nClasses = len(wordIndexes)  # Задаем количество классов выборки
    classesXSamples = []  # Здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна (например, 6 по 1341*1000)"
    for wI in wordIndexes:  # Для каждого текста выборки из последовательности индексов
        classesXSamples.append(getSetFromIndexes(wI, xLen,
                                                 step))  # Добавляем в список очередной текст индексов, разбитый на "кол-во окон*длину окна"

    # Формируем один общий xSamples
    xSamples = []  # Здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна (например, 15779*1000)"
    ySamples = []  # Здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"

    for t in range(nClasses):  # В диапазоне кол-ва классов(6)
        xT = classesXSamples[t]  # Берем очередной текст вида "кол-во окон в тексте*длину окна"(например, 1341*1000)
        for i in range(len(xT)):  # И каждое его окно
            xSamples.append(xT[i])  # Добавляем в общий список выборки
            ySamples.append(utils.to_categorical(t, nClasses))  # Добавляем соответствующий вектор класса

    xSamples = np.array(xSamples)  # Переводим в массив numpy для подачи в нейронку
    ySamples = np.array(ySamples)  # Переводим в массив numpy для подачи в нейронку

    return (xSamples, ySamples)  # Функция возвращает выборку и соответствующие векторы классов


# Представляем тестовую выборку в удобных для распознавания размерах
def createTestMultiClasses(wordIndexes, xLen,
                           step):  # функция принимает последовательность индексов, размер окна, шаг окна

    # Для каждого из 6 классов
    # Создаём тестовую выборку из индексов
    nClasses = len(wordIndexes)  # Задаем количество классов
    xTest6Classes01 = []  # Здесь будет список из всех классов, каждый размером "кол-во окон в тексте * 20000 (при maxWordsCount=20000)"
    xTest6Classes = []  # Здесь будет список массивов, каждый размером "кол-во окон в тексте * длину окна"(6 по 420*1000)
    for wI in wordIndexes:  # Для каждого тестового текста из последовательности индексов
        sample = (
            getSetFromIndexes(wI, xLen, step))  # Тестовая выборка размером "кол-во окон*длину окна"(например, 420*1000)
        xTest6Classes.append(sample)  # Добавляем в список
        xTest6Classes01.append(tokenizer.sequences_to_matrix(
            sample))  # Трансформируется в Bag of Words в виде "кол-во окон в тексте * 20000"
    xTest6Classes01 = np.array(xTest6Classes01)  # И добавляется к нашему списку,
    xTest6Classes = np.array(xTest6Classes)  # И добавляется к нашему списку,

    return xTest6Classes01, xTest6Classes  # функция вернёт тестовые данные: TestBag 6 классов на n*20000 и xTestEm 6 по n*1000


# Распознаём тестовую выборку и выводим результаты
def recognizeMultiClass(model, xTest, modelName):
    print("НЕЙРОНКА: ", modelName)
    print()

    totalSumRec = 0  # Сумма всех правильных ответов

    # Проходим по всем классам
    for i in range(nClasses):
        # Получаем результаты распознавания класса по блокам слов длины xLen
        currPred = model.predict(xTest[i])
        # Определяем номер распознанного класса для каждохо блока слов длины xLen
        currOut = np.argmax(currPred, axis=1)

        evVal = []
        for j in range(nClasses):
            evVal.append(len(currOut[currOut == j]) / len(xTest[i]))

        totalSumRec += len(currOut[currOut == i])
        recognizedClass = np.argmax(evVal)  # Определяем, какой класс в итоге за какой был распознан

        # Выводим результаты распознавания по текущему классу
        isRecognized = "Это НЕПРАВИЛЬНЫЙ ответ!"
        if (recognizedClass == i):
            isRecognized = "Это ПРАВИЛЬНЫЙ ответ!"
        str1 = 'Класс: ' + className[i] + " " * (11 - len(className[i])) + str(
            int(100 * evVal[i])) + "% сеть отнесла к классу " + className[recognizedClass]
        print(str1, " " * (55 - len(str1)), isRecognized, sep='')

    # Выводим средний процент распознавания по всем классам вместе
    print()
    sumCount = 0
    for i in range(nClasses):
        sumCount += len(xTest[i])
    print("Средний процент распознавания ", int(100 * totalSumRec / sumCount), "%", sep='')

    print()

    return totalSumRec / sumCount


className = ["О. Генри", "Стругацкие", "Булгаков", "Саймак", "Фрай", "Брэдберри"]  # Объявляем интересующие нас классы
nClasses = len(className)  # Считаем количество классов

#Загружаем обучающие тексты

trainText = [] #Формируем обучающие тексты
testText = [] #Формируем тестовые тексты

#Формирование необходимо произвести следующим образом
#Класс каждого i-ого эллемента в обучающей выборке должен соответствовать
#классу каждого i-ого эллемента в тестовой выборке

for i in className:
    for j in baseP.glob("*.txt"):
        if i in j.name:
            if 'Обучающая' in j.name: #Если в имени найденного класса есть строка "Обучающая"
               trainText.append(readText(j)) #добавляем в обучающую выборку
               print(j, 'добавлен в обучающую выборку') #Выводим информацию
            if 'Тестовая' in j.name: #Если в имени найденного класса есть строка "Тестовая"
               testText.append(readText(j)) #добавляем в обучающую выборку
               print(j, 'добавлен в тестовую выборку') #Выводим информацию

listMaxWordsCount = [100, 1000, 10000] # count of tokinazer words

exp1 = Table(show_header=True, title="Experiment 1. Change maxWordsCount of tokinazer")
console = Console()
exp1.add_column("maxWordsCount")
exp1.add_column("memoryError")
exp1.add_column("val-acc")
exp1.add_column("Time of tokenization")

for mwc in listMaxWordsCount:
    print(f"maxWorsCount = {mwc}")
    cur_time = time.time()  # Засекаем текущее время

    # Воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
    # num_words=maxWordsCount - определяем максимальное количество слов/индексов, учитываемое при обучении текстов
    # filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' - избавляемся от ненужных символов
    # lower=True - приводим слова к нижнему регистру
    # split=' ' - разделяем слова по пробелу
    # char_level=False - токенизируем по словам (Если будет True - каждый символ будет рассматриваться как отдельный токен )
    tokenizer = Tokenizer(num_words=mwc, filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                          lower=True, split=' ', oov_token='unknown', char_level=False)

    tokenizer.fit_on_texts(
        trainText)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
    tokenTime = round(time.time() - cur_time, 2)
    items = list(tokenizer.word_index.items())  # Вытаскиваем индексы слов для просмотра
    print('Время токенизации: ', tokenTime, 'c', sep='')
    # Преобразовываем текст в последовательность индексов согласно частотному словарю
    trainWordIndexes = tokenizer.texts_to_sequences(trainText)  # Обучающие тесты в индексы
    testWordIndexes = tokenizer.texts_to_sequences(testText)  # Проверочные тесты в индексы

    # Задаём базовые параметры
    xLen = 1000  # Длина отрезка текста, по которой анализируем, в словах
    step = 100  # Шаг разбиения исходного текста на обучающие векторы

    # Формируем обучающую и тестовую выборку
    xTrain, yTrain = createSetsMultiClasses(trainWordIndexes, xLen, step)  # извлекаем обучающую выборку
    xTest, yTest = createSetsMultiClasses(testWordIndexes, xLen, step)  # извлекаем тестовую выборку

    # Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
    xTrain01 = tokenizer.sequences_to_matrix(
        xTrain.tolist())  # П одаем xTrain в виде списка, чтобы метод успешно сработал
    xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())  # Подаем xTest в виде списка, чтобы метод успешно сработал

    # Создаём полносвязную сеть
    model02 = Sequential()
    # Первый полносвязный слой
    model02.add(Dense(200, input_dim=mwc, activation="relu"))
    # Слой регуляризации Dropout
    model02.add(Dropout(0.25))
    # Слой пакетной нормализации
    model02.add(BatchNormalization())
    # Выходной полносвязный слой
    model02.add(Dense(6, activation='softmax'))

    model02.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    with device("CPU:0"):
        # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
        isMemoryError = False

        history = model02.fit(xTrain01,
                              yTrain,
                              epochs=20,
                              batch_size=128,
                              validation_data=(xTest01, yTest))
        model02.compile(optimizer=Adam(0.0001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
        history = model02.fit(xTrain01,
                               yTrain,
                               epochs=10,
                               batch_size=128,
                               validation_data=(xTest01, yTest))


        val_acc = history.history['val_accuracy'][-1]

    exp1.add_row(str(mwc), str(isMemoryError), str(val_acc), str(tokenTime))

exp1.add_row(str(50000), str(True), str(-1), str(-1))


exp2 = Table(show_header=True, title="Experiment 2. Change NN")
exp2.add_column("Num of Layers")
exp2.add_column("Num of Neurons in one layer")
exp2.add_column("Activetion")
exp2.add_column("val-acc")


cur_time = time.time()  # Засекаем текущее время

# Воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
# num_words=maxWordsCount - определяем максимальное количество слов/индексов, учитываемое при обучении текстов
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' - избавляемся от ненужных символов
# lower=True - приводим слова к нижнему регистру
# split=' ' - разделяем слова по пробелу
# char_level=False - токенизируем по словам (Если будет True - каждый символ будет рассматриваться как отдельный токен )
tokenizer = Tokenizer(num_words=mwc, filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                      lower=True, split=' ', oov_token='unknown', char_level=False)

tokenizer.fit_on_texts(
    trainText)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
tokenTime = round(time.time() - cur_time, 2)
items = list(tokenizer.word_index.items())  # Вытаскиваем индексы слов для просмотра
print('Время токенизации: ', tokenTime, 'c', sep='')
# Преобразовываем текст в последовательность индексов согласно частотному словарю
trainWordIndexes = tokenizer.texts_to_sequences(trainText)  # Обучающие тесты в индексы
testWordIndexes = tokenizer.texts_to_sequences(testText)  # Проверочные тесты в индексы

# Задаём базовые параметры
xLen = 1000  # Длина отрезка текста, по которой анализируем, в словах
step = 100  # Шаг разбиения исходного текста на обучающие векторы

# Формируем обучающую и тестовую выборку
xTrain, yTrain = createSetsMultiClasses(trainWordIndexes, xLen, step)  # извлекаем обучающую выборку
xTest, yTest = createSetsMultiClasses(testWordIndexes, xLen, step)  # извлекаем тестовую выборку

# Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
xTrain01 = tokenizer.sequences_to_matrix(
    xTrain.tolist())  # П одаем xTrain в виде списка, чтобы метод успешно сработал
xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())  # Подаем xTest в виде списка, чтобы метод успешно сработал

num_of_layers = [3, 6]
num_of_neurons = [40, 50, 100]
activate_funs = ["relu", 'linear', 'elu']


# Создаём полносвязную сеть
model = Sequential()
# Первый полносвязный слой
model.add(Dense(200, input_dim=mwc, activation="relu"))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
#1
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#2
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#3
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))

# Слой пакетной нормализации
model.add(BatchNormalization())
# Выходной полносвязный слой
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


with device("CPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest))
    model.compile(optimizer=Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=10,
                          batch_size=128,
                          validation_data=(xTest01, yTest))

    val_acc = history.history['val_accuracy'][-1]
    exp2.add_row(str(3), str(40), "relu", str(val_acc))


# Создаём полносвязную сеть
model = Sequential()
# Первый полносвязный слой
model.add(Dense(200, input_dim=mwc, activation="relu"))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
#1
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#2
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#3
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#4
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#5
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))
#6
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.25))

# Слой пакетной нормализации
model.add(BatchNormalization())
# Выходной полносвязный слой
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


with device("CPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest))
    model.compile(optimizer=Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=10,
                          batch_size=128,
                          validation_data=(xTest01, yTest))

    val_acc = history.history['val_accuracy'][-1]
    exp2.add_row(str(6), str(40), "relu", str(val_acc))


# Создаём полносвязную сеть
model = Sequential()
# Первый полносвязный слой
model.add(Dense(200, input_dim=mwc, activation="relu"))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
#1
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))
#2
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))
#3
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))
#4
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))
#5
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))
#6
model.add(Dense(40, activation='linear'))
model.add(Dropout(0.25))

# Слой пакетной нормализации
model.add(BatchNormalization())
# Выходной полносвязный слой
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


with device("CPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest))
    model.compile(optimizer=Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=10,
                          batch_size=128,
                          validation_data=(xTest01, yTest))

    val_acc = history.history['val_accuracy'][-1]
    exp2.add_row(str(6), str(40), "linear", str(val_acc))

# Создаём полносвязную сеть
model = Sequential()
# Первый полносвязный слой
model.add(Dense(200, input_dim=mwc, activation="relu"))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
#1
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#2
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#3
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#4
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))


# Слой пакетной нормализации
model.add(BatchNormalization())
# Выходной полносвязный слой
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


with device("CPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest))
    model.compile(optimizer=Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=10,
                          batch_size=128,
                          validation_data=(xTest01, yTest))

    val_acc = history.history['val_accuracy'][-1]
    exp2.add_row(str(4), str(100), "elu", str(val_acc))

model = Sequential()
# Первый полносвязный слой
model.add(Dense(200, input_dim=mwc, activation="relu"))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
#1
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#2
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#3
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))
#4
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))

#5
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))

#6
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.25))

# Слой пакетной нормализации
model.add(BatchNormalization())
# Выходной полносвязный слой
model.add(Dense(6, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


with device("CPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest))
    model.compile(optimizer=Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model.fit(xTrain01,
                          yTrain,
                          epochs=10,
                          batch_size=128,
                          validation_data=(xTest01, yTest))

    val_acc = history.history['val_accuracy'][-1]
    exp2.add_row(str(6), str(100), "elu", str(val_acc))


exp3 = Table(show_header=True, title="Experiment 3. Change Embedding len")
exp3.add_column("Len")
exp3.add_column("Val-acc")

maxWordsCount = 10000

lens = [10, 50, 200]

for l in lens:
    model03 = Sequential()
    model03.add(Embedding(maxWordsCount, l, input_length=xLen))
    model03.add(SpatialDropout1D(0.2))
    model03.add(Flatten())
    model03.add(BatchNormalization())
    model03.add(Dense(200, activation="relu"))
    model03.add(Dropout(0.2))
    model03.add(BatchNormalization())
    model03.add(Dense(6, activation='softmax'))
    model03.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    with device("CPU:0"):
        history = model03.fit(xTrain,
                              yTrain,
                              epochs=10,
                              batch_size=128,
                              validation_data=(xTest, yTest))
        exp3.add_row(str(l), str(history.history['val_accuracy'][-1]))

console.print(exp1)
console.print(exp2)
console.print(exp3)