import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Softmax, Conv2D, MaxPool2D, Flatten, ReLU, BatchNormalization
from keras.utils import to_categorical
from nnom import generate_model, evaluate_model

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(test_images.shape)
train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

inputs = Input(shape=(28, 28, 1,))
# x = Dense(128, activation='relu')(inputs)
# x = Dense(32, activation='relu')(x)

x = Conv2D(32, kernel_size=(5, 5))(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

# x = Conv2D(16, kernel_size=(5, 5))(x)
# x = BatchNormalization()(x)
# x = ReLU()(x)
# x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

x = Flatten()(x)
x = Dense(32)(x)
x = ReLU()(x)
x = Dense(10)(x)

outputs = Softmax()(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=1, batch_size=32, verbose=1)
model.save('test.h5')

generate_model(model, test_images, name='weights.h', format='hwc', quantize_method='kld')

# evaluate_model(model, x_test=test_images, y_test=test_labels)