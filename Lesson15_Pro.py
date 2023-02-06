from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
#import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def invert_image(im):
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            im[x][y] = abs(im[x][y]-255)

#create and train nn with best parameters founded in Lite-level
f = "relu"
nc = 5000
bs = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # load mnist dataset

# change image format
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# normal the images
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255
# Преобразуем ответы в формат one_hot_encoding
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Dense(nc, input_dim=784, activation=f))
model.add(Dense(400, activation=f))
model.add(Dense(10, activation="softmax"))
# compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=bs, epochs=15, verbose=1)

#load my numbers
p1 = image.load_img(path="./resouces/1.png", target_size=(28,28), color_mode="grayscale")
p5 = image.load_img(path="./resouces/5.png", target_size=(28,28), color_mode="grayscale")

p1 = image.img_to_array(p1)
p5 = image.img_to_array(p5)

#dim extra axes
p1 = p1.squeeze()
p5 = p5.squeeze()

#print(p1, p5)
#plt.imshow(p1)
# plt.show()
# plt.imshow(p5)
# plt.show()

invert_image(p1)
invert_image(p5)
# print(p1, p5)
# plt.imshow(p1)
# plt.show()
# plt.imshow(p5)
# plt.show()


# image to nn format
p1 = p1.reshape(1, 784)
p5 = p5.reshape(1, 784)
#print(p1.shape, p5.shape)
p1 = p1.astype('float32')
p5 = p5.astype('float32')
p1 /= 255
p5 /= 255

#predict
print("################")
pred1 = model.predict(p1)
print(np.argmax(pred1))

pred5 = model.predict(p5)
print(np.argmax(pred5))