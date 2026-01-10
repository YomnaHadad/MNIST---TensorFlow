{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPCMUMAQ3+p2mItXf4QWael",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/YomnaHadad/MNIST---TensorFlow/blob/main/TF_mnist_cnn_py.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## TensorFlow - Computer Vision - Mnist\n"
      ],
      "metadata": {
        "id": "Qy7FyGKOy8N2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow\n",
        "from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAvgPool2D, BatchNormalization\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "model = tensorflow.keras.Sequential(\n",
        "    [\n",
        "        Input(shape=(28,28,1)),\n",
        "        Conv2D(32, (3,3), activation='relu'),   # expects 4 dims\n",
        "        MaxPooling2D(),\n",
        "        BatchNormalization(),\n",
        "        Conv2D(64, (3,3), activation='relu'),\n",
        "        MaxPooling2D(),\n",
        "        BatchNormalization(),\n",
        "\n",
        "        GlobalAvgPool2D(),\n",
        "        Dense(64, activation='relu'),\n",
        "        Dense(10, activation='softmax')\n",
        "    ]\n",
        ")\n",
        "\n",
        "\n",
        "def display_samples(examples, labels):\n",
        "  '''create a figure of randomly-choosen data samples from the downloaded dataset '''\n",
        "  fig = plt.figure(figsize=(10,10))\n",
        "  for i in range(25):\n",
        "    idx = np.random.randint(examples.shape[0])   # index of random feature\n",
        "    img = examples[idx]\n",
        "    img_label = labels[idx]\n",
        "    plt.subplot(5,5,i+1)\n",
        "    plt.imshow(img, cmap='gray')\n",
        "    # plt.title(img_label)\n",
        "  plt.show()\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()\n",
        "    print(\"shape of x_train: \", x_train.shape)\n",
        "    print(\"shape of y_train: \", y_train.shape)\n",
        "    print(\"shape of x_test: \", x_test.shape)\n",
        "    print(\"shape of y_test: \", y_test.shape)\n",
        "\n",
        "    # display_samples(x_train, y_train)\n",
        "\n",
        "    x_train = x_train.astype(float) / 255.\n",
        "    x_test = x_test.astype(float) / 255.\n",
        "\n",
        "    # expand the dimensions to match the input shape (28, 28, 1)\n",
        "    x_train = np.expand_dims(x_train, axis=-1)\n",
        "    x_test = np.expand_dims(x_test, axis=-1)\n",
        "\n",
        "    # Hot-Encoding\n",
        "    # y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)\n",
        "    # y_test = tensorflow.keras.utils.to_catorgorical(y_test, num_classes=10)\n",
        "    # model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "    model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy' , metrics=['accuracy'])\n",
        "    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=.2)\n",
        "    model.evaluate(x_test, y_test)\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GulXdNTA07i1",
        "outputId": "0dcc58f4-c679-40dd-e584-5e66ee55e1df"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "shape of x_train:  (60000, 28, 28)\n",
            "shape of y_train:  (60000,)\n",
            "shape of x_test:  (10000, 28, 28)\n",
            "shape of y_test:  (10000,)\n",
            "Epoch 1/3\n",
            "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 6ms/step - accuracy: 0.7648 - loss: 0.8140 - val_accuracy: 0.9390 - val_loss: 0.2058\n",
            "Epoch 2/3\n",
            "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 4ms/step - accuracy: 0.9710 - loss: 0.0980 - val_accuracy: 0.9715 - val_loss: 0.0898\n",
            "Epoch 3/3\n",
            "\u001b[1m750/750\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 4ms/step - accuracy: 0.9803 - loss: 0.0656 - val_accuracy: 0.9656 - val_loss: 0.1114\n",
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 6ms/step - accuracy: 0.9581 - loss: 0.1325\n"
          ]
        }
      ]
    }
  ]
}
