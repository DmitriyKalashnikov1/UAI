import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential  # Полносвязная модель
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, \
    Activation  # Слои для сети
from tensorflow.keras.preprocessing.text import \
    Tokenizer  # Методы для работы с текстами и преобразования их в последовательности

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import device
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

# Загружаем обучающие тексты

trainText = []  # Формируем обучающие тексты
testText = []  # Формируем тестовые тексты



# Формирование необходимо произвести следующим образом
# Класс каждого i-ого эллемента в обучающей выборке должен соответствовать
# классу каждого i-ого эллемента в тестовой выборке

for i in className:
    for j in baseP.glob("*.txt"):
        if i in j.name:
            if 'Обучающая' in j.name:  # Если в имени найденного класса есть строка "Обучающая"
                trainText.append(readText(j))  # добавляем в обучающую выборку
                print(j, 'добавлен в обучающую выборку')  # Выводим информацию
            if 'Тестовая' in j.name:  # Если в имени найденного класса есть строка "Тестовая"
                testText.append(readText(j))  # добавляем в обучающую выборку
                print(j, 'добавлен в тестовую выборку')  # Выводим информацию

cur_time = time.time()  # Засекаем текущее время
maxWordsCount = 20000  # Определяем максимальное количество слов/индексов, учитываемое при обучении текстов

# Воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
# num_words=maxWordsCount - определяем максимальное количество слов/индексов, учитываемое при обучении текстов
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' - избавляемся от ненужных символов
# lower=True - приводим слова к нижнему регистру
# split=' ' - разделяем слова по пробелу
# char_level=False - токенизируем по словам (Если будет True - каждый символ будет рассматриваться как отдельный токен )
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                      lower=True, split=' ', oov_token='unknown', char_level=False)

tokenizer.fit_on_texts(
    trainText)  # "Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности
items = list(tokenizer.word_index.items())  # Вытаскиваем индексы слов для просмотра
print('Время токенизации: ', round(time.time() - cur_time, 2), 'c', sep='')

# print(items)
print("Размер словаря", len(items))  # Длина словаря

# Преобразовываем текст в последовательность индексов согласно частотному словарю
trainWordIndexes = tokenizer.texts_to_sequences(trainText)  # Обучающие тесты в индексы
testWordIndexes = tokenizer.texts_to_sequences(testText)  # Проверочные тесты в индексы

print("Взглянем на фрагмент обучающего текста:")
print("В виде оригинального текста:              ", trainText[1][:87])
print("Он же в виде последовательности индексов: ", trainWordIndexes[1][:20], '\n')

# Задаём базовые параметры
xLen = 1000  # Длина отрезка текста, по которой анализируем, в словах
step = 100  # Шаг разбиения исходного текста на обучающие векторы

cur_time = time.time()  # Засекаем текущее время
# Формируем обучающую и тестовую выборку
xTrain, yTrain = createSetsMultiClasses(trainWordIndexes, xLen, step)  # извлекаем обучающую выборку
xTest, yTest = createSetsMultiClasses(testWordIndexes, xLen, step)  # извлекаем тестовую выборку

# получили обучающий/тестовый набор, достаточный для запуска Embedding, но для Bag of Words нужно xTrain и xTest представить в виде векторов из 0 и 1
print('Время обработки Embedding: ', round(time.time() - cur_time, 2), 'c', sep='')
cur_time = time.time()  # Засекаем текущее время
# Преобразовываем полученные выборки из последовательности индексов в матрицы нулей и единиц по принципу Bag of Words
xTrain01 = tokenizer.sequences_to_matrix(xTrain.tolist())  # П одаем xTrain в виде списка, чтобы метод успешно сработал
xTest01 = tokenizer.sequences_to_matrix(xTest.tolist())  # Подаем xTest в виде списка, чтобы метод успешно сработал

print('Время обработки BoW: ', round(time.time() - cur_time, 2), 'c', sep='')

xTest6Classes01, x2 = createTestMultiClasses(testWordIndexes, xLen, step)  # Преобразование тестовой выборки

checkpoint = ModelCheckpoint(filepath="./outputs/bestLesson18Pro.hdf5", save_freq="epoch", save_best_only=True, save_weights_only=False)

# Создаём полносвязную сеть
model02 = Sequential()
# Первый полносвязный слой
model02.add(Dense(150, input_dim=maxWordsCount, activation="relu"))
# Слой регуляризации Dropout
model02.add(Dropout(0.25))
# Слой пакетной нормализации
#model02.add(BatchNormalization())
model02.add(Dense(150,activation="relu"))
# Слой регуляризации Dropout
model02.add(Dropout(0.25))
# Слой пакетной нормализации
model02.add(BatchNormalization())
# Выходной полносвязный слой
model02.add(Dense(6, activation='softmax'))

model02.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

BoWacc = []
BoWval_acc = []

with device("/GPU:0"):
    # Обучаем сеть на выборке, сформированной по bag of words - xTrain01
    history = model02.fit(xTrain01,
                          yTrain,
                          epochs=20,
                          batch_size=128,
                          validation_data=(xTest01, yTest),
                          callbacks=[checkpoint])
    BoWacc += history.history['accuracy']
    BoWval_acc += history.history['val_accuracy']



model02 = load_model("bestLesson18Pro.hdf5")

#Проверяем точность нейронки обученной на bag of words
pred = recognizeMultiClass(model02, xTest6Classes01, "Тексты 01 + Dense")
plt.figure(figsize=(14,7))
plt.plot(BoWacc,
         label='Доля верных ответов на обучающем наборе (BoW)')
plt.plot(BoWval_acc,
         label='Доля верных ответов на проверочном наборе (BoW)')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()