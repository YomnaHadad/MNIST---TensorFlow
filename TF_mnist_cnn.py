import tensorflow
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAvgPool2D, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),   # expects 4 dims
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)


def display_samples(examples, labels):
  '''create a figure of randomly-choosen data samples from the downloaded dataset '''
  fig = plt.figure(figsize=(10,10))
  for i in range(25):
    idx = np.random.randint(examples.shape[0])   # index of random feature
    img = examples[idx]
    img_label = labels[idx] 
    plt.subplot(5,5,i+1)
    plt.imshow(img, cmap='gray')
    # plt.title(img_label)
  plt.show()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    print("shape of x_train: ", x_train.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of x_test: ", x_test.shape)
    print("shape of y_test: ", y_test.shape)

    # display_samples(x_train, y_train)

    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.
    
    # expand the dimensions to match the input shape (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Hot-Encoding
    # y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)
    # y_test = tensorflow.keras.utils.to_catorgorical(y_test, num_classes=10)
    # model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy' , metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=.2)
    model.evaluate(x_test, y_test)


