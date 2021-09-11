import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Activation, Flatten, Dropout, Dense, MaxPooling2D
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=DeprecationWarning)

#Conv2D — сверточный слой
#Activation — слой активации
#MaxPooling2D -слой дискретизации на основе выборки.
#   Цель состоит в том, чтобы уменьшить выборку входного представления (изображение, выходную матрицу скрытого слоя и т. Д.),
#   Уменьшив его размерность и сделав допущения относительно предположений о свойствах, содержащихся в выбранных субрегионах.
#   Работает на основе выбора максимального значения из подвыборки размером pool_size.
#Dropout — слой прореживания для решения проблемы переобучения сети
#Flatten — преобразование в одномерный вектор
#Dense — полносвязный слой


class Running:
    @classmethod
    def train(cls, num_epochs):
        train_answers = np.zeros((6600, 10))
        test_answers = np.zeros((2200, 10))
        train_answers = cls.fill_answers(train_answers, 660)
        test_answers = cls.fill_answers(test_answers, 220)

        print("Обучение...")
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(93, 13, 1)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, kernel_size=3, activation='relu'))
        model.add(Flatten())  #– слой, преобразующий 2D-данные в 1D-данные.
        model.add(Dense(10, activation='softmax'))
        #############################################

        # optimizer = 'adam'(Адам: метод стохастической оптимизации).Функция потерь: loss = 'categorical_crossentropy'
        # категориальная ерекрестная энтропия(categorical crossentropy CCE)
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

        history = model.fit(listTrain, train_answers, epochs=num_epochs, validation_data=(listTest, test_answers))

        print("История:")
        print(history.history)
        model.save('Model')
        cls.graph(history)

    @classmethod
    def fill_answers(cls, answers_list, data_len):
        for j in range(0, 10):
            for i in range(0, data_len):
                val = np.zeros(10)
                val[j] = 1
                answers_list[j * data_len + i] = val
        return answers_list

    @classmethod
    def graph(cls, history):
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Модель Точность')
        plt.ylabel('Точность')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
        #
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Модель потерь')
        plt.ylabel('Потеря')
        plt.xlabel('Эпохи')
        plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
        plt.show()

    @classmethod
    def test(cls, listTest):
        print("Тест...")
        model = tf.keras.models.load_model('Model')
        test_answers = np.zeros((2200, 10))
        test_answers = cls.fill_answers(test_answers, 220)
        score = model.evaluate(listTest, test_answers, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
