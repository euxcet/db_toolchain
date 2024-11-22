from keras.datasets import mnist
import numpy as np
import keras
model = keras.models.load_model('test.h5')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images[0], train_labels[0])

x = np.zeros((1, 784))

for i in range(5, 20):
    x[0][i * 28 + 14] = 255
    x[0][i * 28 + 15] = 255
    x[0][i * 28 + 16] = 255

print(x)

y = model.predict(x)
print(y)
