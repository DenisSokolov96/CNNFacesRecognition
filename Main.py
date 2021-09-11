import PySimpleGUI as sg
import numpy as np

from Running import *

def main():
    #Running.train(listTrain, listTest, 1)
    #Running.test(listTest)
    history = None
    num_epochs = 1
    layout = [
        [sg.Button("Обучить"), sg.Button("Распознать")],
        [sg.Text("Эпохи"), sg.Input(key='-INPUT-', default_text="1", size=(7, 1), justification='center')],
        [sg.Output(size=(88, 20), key='out')]
    ]
    window = sg.Window('CNN', layout)
    while True:
        event, values = window.read()
        if event == "Обучить" or event == "Распознать":
            try:
                num_epochs = int(values['-INPUT-'])
            except ValueError:
                print("Ошибка значения, выполняется для:" + str(num_epochs))
        if event == "Обучить":
            try:
                Running.train(num_epochs)
            except Exception as inst:
                print(inst)
        elif event == "Распознать":
            Running.test(listTest)
        if event in (sg.WIN_CLOSED, 'Quit'):
            break
    window.close()


if __name__ == '__main__':
        main()
