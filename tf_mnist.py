import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

input_size = 784

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape =(28,28,1)),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    print("shape of x_train: ", x_train.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of x_test: ", x_test.shape)
    print("shape of y_test: ", y_test.shape)

    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.

    # expand the dimensions to match the input shape (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    print(model.summary())
  
    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy' , metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=.2)
    model.evaluate(x_test, y_test)


