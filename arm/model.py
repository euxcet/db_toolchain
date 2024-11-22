import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Softmax
from keras.utils import to_categorical
from nnom import generate_model, evaluate_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(test_images.shape)
size = 784
train_images = np.reshape(train_images, (-1, 784))[:, :size]
test_images = np.reshape(test_images, (-1, 784))[:, :size]
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

print(test_images.shape)

inputs = Input(shape=(size,))
# x = Dense(128, activation='relu')(inputs)
# x = Dense(32, activation='relu')(x)
x = Dense(10)(inputs)

outputs = Softmax()(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
model.save('test.h5')

generate_model(model, test_images, name='weights.h', format='hwc', quantize_method='kld')

evaluate_model(model, x_test=test_images, y_test=test_labels)
