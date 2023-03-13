import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential  # Полносвязная модель
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, \
    Conv1D, LSTM, Bidirectional # Слои для сети
from tensorflow.keras.preprocessing.text import \
    Tokenizer  # Методы для работы с текстами и преобразования их в последовательности

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import device
from pathlib import Path
import time

inCollab = False

if inCollab:
    from google.colab import drive

    drive.mount('/content/drive')

    baseP = "/content/drive/MyDrive/resouces/illness"
    pathToGraph1 = ""
    pathToGraph2 = ""
    pathToGraph3 = ""
    pathToGraph4 = ""
else:
    baseP = Path('.') / 'resouces' / 'Illness/'
    pathToGraph1 = "./outputs/Lesson19ULModelEmbAndSimpDense.png"
    pathToGraph2 = "./outputs/Lesson19ULModelEmbAndLSTM.png"
    pathToGraph3 = "./outputs/Lesson19ULModelEmbAndConv.png"
    pathToGraph4 = "./outputs/Lesson19ULModelComplex.png"


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
    xTest6Classes01 = np.array(xTest6Classes01, dtype='object')  # И добавляется к нашему списку,
    xTest6Classes = np.array(xTest6Classes, dtype='object')  # И добавляется к нашему списку,

    return xTest6Classes01, xTest6Classes  # функция вернёт тестовые данные: TestBag 6 классов на n*20000 и xTestEm 6 по n*1000


# Распознаём тестовую выборку и выводим результаты
def recognizeMultiClass(model, xTest, modelName, className):
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

# Создаем функцию, которая отключает блоки по индексу размером окна window
def block_off(X, i, window):
    return np.where((i*window < X) & (X <= (i+1)*window), 1, X)




className = set()  # Объявляем интересующие нас классы

for j in baseP.glob("*.txt"):
    #print(j.name)
    name = j.name.replace(".txt", '')
    name = name.replace("Обучающая", '')
    name = name.replace("Тестовая", '')
    className.add(name)
print(f"Болезни: {className}")
nClasses = len(className)  # Считаем количество классов

# Загружаем обучающие тексты

trainText = []  # Формируем обучающие тексты
testText = []  # Формируем тестовые тексты



# Формирование необходимо произвести следующим образом
# Класс каждого i-ого эллемента в обучающей выборке должен соответствовать
# классу каждого i-ого эллемента в тестовой выборке
#
for i in className:
    for j in baseP.glob("*.txt"):
        if i in j.name:
            if 'Обучающая' in j.name:  # Если в имени найденного класса есть строка "Обучающая"
                trainText.append(readText(j))  # добавляем в обучающую выборку
                print(j, 'добавлен в обучающую выборку')  # Выводим информацию
            if 'Тестовая' in j.name:  # Если в имени найденного класса есть строка "Тестовая"
                testText.append(readText(j))  # добавляем в обучающую выборку
                print(j, 'добавлен в тестовую выборку')  # Выводим информацию

# #################
# # Преобразовываем текстовые данные в числовые/векторные для обучения нейросетью
# #################

maxWordsCount = 50000 # определяем макс.кол-во слов/индексов, учитываемое при обучении текстов

# для этого воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
tokenizer = Tokenizer(num_words=maxWordsCount, filters='–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff', lower=True, split=' ', char_level=False, oov_token = 'unknown')
# выше задаем параметры:
# (num_words=maxWordsCount) - определяем макс.кол-во слов/индексов, учитываемое при обучении текстов
# (filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') - избавляемся от ненужных символов
# (lower=True) - приводим слова к нижнему регистру
# (split=' ') - разделяем слова по пробелу
# (char_level=False) - просим токенайзер не удалять однобуквенные слова
print("Fit tokenzer...")
tokenizer.fit_on_texts(trainText) #"скармливаем" наши тексты, т.е даём в обработку методу, который соберет словарь частотности
print("Done!")
# преобразовываем текст в последовательность индексов согласно частотному словарю
trainWordIndexes = tokenizer.texts_to_sequences(trainText) # обучающие тесты в индексы
testWordIndexes = tokenizer.texts_to_sequences(testText)   # проверочные тесты в индексы
# Задаём базовые параметры
xLen = 65 # Длина отрезка текста, по которой анализируем, в словах
step = 1 # Шаг разбиения исходного текста на обучающие вектора


#create train&test sets
x_train, y_train = createSetsMultiClasses(trainWordIndexes, xLen, step)
x_test, y_test = createSetsMultiClasses(testWordIndexes, xLen, step)

xTest6Classes01, xTest6Classes = createTestMultiClasses(testWordIndexes, xLen, step) # подгоним форму тестовых классов под функцию recognizeMultiClass


with device("/device:CPU:0"):
    model = Sequential()
    model.add(Embedding(maxWordsCount, 70, input_length=xLen))
    model.add(SpatialDropout1D(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(
        LSTM(8, return_sequences=True)))  # добавляем слой с двунаправленным LSTM, совместимый с Cuda при поддержке GPU
    model.add(Bidirectional(
        LSTM(8, return_sequences=True)))  # добавляем слой с двунаправленным LSTM, совместимый с Cuda при поддержке GPU
    model.add(Dropout(
        0.3))  # добавляем слой регуляризации, "выключая" указанное количество нейронов, во избежание переобучения
    model.add(BatchNormalization())  # добавляем слой нормализации данных
    model.add(Flatten())
    #model.add(Dense(600, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    # Компиляция, составление модели с выбором алгоритма оптимизации, функции потерь и метрики точности
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Выводим summary модели

    history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    validation_data=(x_test, y_test))

    pred = recognizeMultiClass(model, xTest6Classes, "EmbAndSimpDense", list(className)) #функция покажет какие классы и как распознаны верно
# Запоминаем точность, с которой будем сравнивать наши блоки
normal_accuracy = model.evaluate(x_test, y_test)[1]
print("Normal accuracy: " + str(normal_accuracy))
window = 100  # Задаем размер блока отключаемых токенов
data = []  # Заготовка для таблицы точности модели
for i in range(maxWordsCount // window):  # Проходим по всем токенам с шагом 100

    print(i, '===' * 30)
    x_testST2 = block_off(X=x_test, i=i, window=window)  # Формируем новый test с отключенными токенами

    scorr = model.evaluate(x_testST2, y_test)[1]

    data += [[round(scorr, 3)]]

data_1 = pd.DataFrame(data) # DF индекс блока и точность
data_del = data_1.index[data_1[0] > normal_accuracy] # Отбираем только те индексы блоков, которые хотим отключить

ind_off = []                # Токены, которые нужно отключить
for i in data_del:        # Проходим по индексам блоков
    ind_off.extend(range(i*window, (i+1)*window))  # Добавляем токены, которые будем отключать
print("Tokens to turn off: "+str(ind_off))
train = x_train.copy()
for i in ind_off: # Проходим по списку токенов, которые нужно заменить на 1
    train = np.where(train == i, 1, train) # Формируем новый train

with device("/device:CPU:0"):
    model = Sequential()
    model.add(Embedding(maxWordsCount, 70, input_length=xLen))
    model.add(SpatialDropout1D(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(
        LSTM(8, return_sequences=True)))  # добавляем слой с двунаправленным LSTM, совместимый с Cuda при поддержке GPU
    model.add(Bidirectional(
        LSTM(8, return_sequences=True)))  # добавляем слой с двунаправленным LSTM, совместимый с Cuda при поддержке GPU
    model.add(Dropout(
        0.3))  # добавляем слой регуляризации, "выключая" указанное количество нейронов, во избежание переобучения
    model.add(BatchNormalization())  # добавляем слой нормализации данных
    model.add(Flatten())
    #model.add(Dense(600, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    # Компиляция, составление модели с выбором алгоритма оптимизации, функции потерь и метрики точности
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary() # Выводим summary модели

    history = model.fit(train,
                    y_train,
                    epochs=100,
                    batch_size=512,
                    validation_data=(x_test, y_test))

test_accuracy = model.evaluate(x_test, y_test, verbose = 0)[1]

print('До отключения токенов точность была', normal_accuracy)
print('После                              ', test_accuracy)