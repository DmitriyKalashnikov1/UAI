import numpy as np # Для работы с данными
import pandas as pd # Для работы с таблицами
import matplotlib.pyplot as plt # Для вывода графиков
import os # Для работы с файлами


from tensorflow.keras import utils # Для работы с категориальными данными
from tensorflow.keras.models import Sequential # Полносвязная модель
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation # Слои для сети
from tensorflow.keras.preprocessing.text import Tokenizer # Методы для работы с текстами и преобразования их в последовательности
from tensorflow.keras.preprocessing.sequence import pad_sequences # Метод для работы с последовательностями

from sklearn.preprocessing import LabelEncoder # Метод кодирования тестовых лейблов
from sklearn.model_selection import train_test_split # Для разделения выборки на тестовую и обучающую


def word_func(arg, edge):
    '''
    Input:
        arg - Словарь частнотности слов
        edge - Край, по которому будут выбраны все слова, кол-во символов которых больше числа edge

    Return:
        indexes - индексы в порядке возрастания значений arg и тех, кто прошёл порог(edge).
        '''
    indexes = []  # Здесь будут индексы
    uniq_nums = sorted(set(arg.values()), reverse=True)  # Получим все уникальные значения arg словаря
    all_nums = list(arg.values())  # Получим все значения arg словаря
    for i in uniq_nums:  # Проходимся по уникальным значениям
        for idx, n in enumerate(
                all_nums):  # Проходимся по всем элементам значеням arg словаря и не забываем уточнить их индексы
            if n == i and idx != 0 and len(tokenizer.index_word[
                                               idx]) > edge:  # Проверка на то, что текущее значение из всех элементов равно тому, по которому делаем фильтр(i), проверяем, чтобы индексы не был равен 0
                # Иначе может быть ошибка. Так же проверяем на порог.
                indexes.append(idx)
    return indexes

inCollab = False
if inCollab:
    from google.colab import drive

    drive.mount('/content/drive')
    path = "/content/drive/MyDrive/resouces/psy.csv"
else:
    path = "./resouces/psy.csv"

#считываем базу
base = pd.read_csv(open(path, errors='replace'))

#clear database
base.drop(base.columns[[0,1,2]], axis=1, inplace=True)
base.iloc[:,1] = base.iloc[:,1].astype('uint8')
#get train data
x_data = base.iloc[:,0].values
y_data = base.iloc[:,1].values
#print(x_data, y_data)
x_data_classes = [''.join(x_data[y_data==0])]
x_data_classes.append(''.join(x_data[y_data==1]))

y_data = utils.to_categorical(y_data,2, dtype='uint8')

data = ''.join(x_data)  # В первую строку пойдут все строки

num = 0.3  # Порог 30 процентов по символам в x_train и x_test
idx_data = int(len(x_data)*num)

X_Train = x_data[:-idx_data]  # Берём данные для обучающей выборки.
y_train = y_data[:-idx_data]

X_Test = x_data[-idx_data:]  # Берёи данные для проверочной выборки.
y_test = y_data[-idx_data:]

#print(X_Train, y_train)

#тонизация и превращение в bow
max_words_count = 3000
tokenizer = Tokenizer(num_words=max_words_count, filters='–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff',
                      lower=True, split=' ', char_level=False, oov_token='unknown')
tokenizer.fit_on_texts(x_data)

# Воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
# num_words=maxWordsCount - определяем максимальное количество слов/индексов, учитываемое при обучении текстов
# filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' - избавляемся от ненужных символов
# lower=True - приводим слова к нижнему регистру
# split=' ' - разделяем слова по пробелу
# char_level=False - токенизируем по словам (Если будет True - каждый символ будет рассматриваться как отдельный токен )

for_hist_nums = np.array(tokenizer.texts_to_sequences(x_data_classes))  # Для гистограмки

X_train = tokenizer.texts_to_sequences(X_Train)  # Преобразования в индексы
X_test = tokenizer.texts_to_sequences(X_Test)

xlen = 15
x_train_em = pad_sequences(X_train, padding='post', maxlen=xlen)  # Преобразования в матрицу индексов
x_test_em = pad_sequences(X_test, padding='post', maxlen=xlen)

# #Для BoW
x_train_bow = tokenizer.sequences_to_matrix(x_train_em.tolist()).astype('uint8')  # Преобразования в BoW
x_test_bow = tokenizer.sequences_to_matrix(x_test_em.tolist()).astype('uint8')
#print(x_train_bow)

#нейронка
model = Sequential()
model.add(Dense(150, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(600, activation='relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train_bow, y_train, validation_data=(x_test_bow, y_test), epochs = 35, batch_size = 200)
print("Максимум точности который можно получить от данной нейронки используя колбеки: ",max(history.history['val_accuracy']))

needPlot = False

if needPlot:
    plt.plot(history.history['accuracy'], label = 'Обучающая')
    plt.plot(history.history['val_accuracy'], label = 'Проверочная')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()
    plt.plot(history.history['loss'], label = 'Обучающая')
    plt.plot(history.history['val_loss'], label = 'Проверочная')
    plt.title('Ошибка')
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()

predictions = np.array([np.argmax(i) for i in model.predict(x_test_bow)])
y_test_real = np.array([np.argmax(i) for i in y_test])
for i in range(20,34):
    print(f'Предсказано - {predictions[i]}, было {y_test_real[i]},  {predictions[i]==y_test_real[i]}')
print(f'\n Процент верных предсказаний - {round((predictions == y_test_real).mean()*100,2)} %')

#гистограмма
for i in range(len(for_hist_nums)):
    for_hist_nums[i] = np.array(for_hist_nums[i])  # Превращаем листы в numpy массивы для работы без ошибок

edge = 50  # Граница кол-ва частотных слов
for_edge = 10  # Граница просмотра первых for_edge слов.
for j in range(2):  # Проходимся по всем классам(2)
    nums = {i: len(for_hist_nums[j][for_hist_nums[j] == i]) for i in
            range(edge)}  # Создаём текущий словарь {индексов : кол-ва слов} в данном классе.
    for edge_min in (0, 3):  # Сначала он пройдётся по границе 0, потом по границе 3
        indexes = word_func(nums, edge_min)  # Получем отсортированный список индексов
        print(f'\n{for_edge} частых слова длинее {edge_min}-ёх символов: \n')
        for g in indexes[:for_edge]:
            print(tokenizer.index_word[g], round(100 * nums[g] / sum(nums.values()), 2),
                  '%')  # Выводим слово и его процентное соотношение
    plt.bar(range(0, 50), list(nums.values()))
    plt.title('Не спам' if j == 0 else 'Спам')
    plt.xlabel('Индекс частых слов')
    plt.ylabel('Кол-во этих слов в выборке')
    plt.show()