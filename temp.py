import numpy as np
import pandas as pd
from math import *


def sigmoid(x, a=1., b=0.):
    return a / (1. + np.exp(-x)) + b


def sigmoid_prime(x, a=1., b=1.):
    return a * sigmoid(x) * (1. - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prine(x):
    return 1. - tanh(x) ** 2


def softsign(x):
    return x / (1. + abs(x))


def softsign_prime(x):
    return 1. / ((1. + abs(x)) ** 2)


def step_function(x):
    return 0 if x < 0 else 1.


def step_function_prime(x):
    return 0


def softplus(x):
    return np.log(1. + np.exp(x))


def softplus_prime(x):
    return sigmoid(x)


def relu(x):
    return np.maximum(x, 0)


def relu_prime(x):
    return 1. * (x > 0)


def pRelu(param, x):
    return param * x if x < 0 else x


def pRelu_prime(param, x):
    return param if x < 0 else 1


def elu(param, x):
    return param * (np.eps(x) - 1.) if x < 0 else x


def elu_prime(param, x):
    return elu(param, x) + param if x < 0 else 1.


def mse(y, y_pred):
    return 0.5 * (np.sum(y_pred - y) ** 2)


def cross_entropy(y, y_pred):
    return -1. / y.size() * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


class neural_network:
    layouts = None
    activation_func = None
    target_function = None
    learning_rate = None
    w = None
    b = None

    def __init__(self, layouts, activation_func, target_func, learning_rate=0.01):
        self.layouts = layouts
        self.activation_func = activation_func
        self.activation_prime = globals()[str(activation_func.__name__) + '_prime']
        self.target_func = target_func
        self.learning_rate = learning_rate
        pass

    def recreate_model(self, layouts=None, activation_func=None, target_func=None, learning_rate=0.01):
        if layouts is None:
            self.layouts = layouts
        if activation_func is None:
            self.activation_func = activation_func
            self.activation_prime = globals()[str(activation_func.__name__) + '_prime']
        if target_func is None:
            self.target_func = target_func
        if learning_rate != 0.01:
            self.learning_rate = learning_rate
        pass

    def bias_init(self, normal=False, variance=0.1):
        if not normal:
            self.b = np.zeros(len(self.layouts) - 1)
        else:
            self.b = np.random.normal(1, len(self.layouts) - 1, (0, variance))
        pass

    def weight_init(self, uniform=True, normal=False, glorot=False, he=False):
        self.w = list()
        if uniform:
            if glorot:
                for i in range(0, len(self.layouts) - 1):
                    self.w.append(np.random.uniform(-sqrt(6) / sqrt(self.layouts[i] + self.layouts[i + 1]),
                                                    sqrt(6) / sqrt(self.layouts[i] + self.layouts[i + 1]),
                                                    (self.layouts[i], self.layouts[i + 1])))
            else:
                for i in range(0, len(self.layouts) - 1):
                    self.w.append(np.random.uniform(0, sqrt(2 / self.layouts[i]),
                                                    (self.layouts[i], self.layouts[i + 1])))
        else:
            if glorot:
                for i in range(0, len(self.layouts) - 1):
                    self.w.append(np.random.normal(-sqrt(6) / sqrt(self.layouts[i] + self.layouts[i + 1]),
                                                   sqrt(6) / sqrt(self.layouts[i] + self.layouts[i + 1]),
                                                   (self.layouts[i], self.layouts[i + 1])))
            else:
                for i in range(0, len(self.layouts) - 1):
                    self.w.append(np.random.normal(0, sqrt(2 / self.layouts[i]),
                                                   (self.layouts[i], self.layouts[i + 1])))

    def propagation(self, outputs):
        for i in range(len(self.layouts) - 1):
            outputs.append(self.activation_func(np.dot(outputs[i], self.w[i]) + self.b[i]))
        return outputs

    def back_propagation(self, outputs, correct):
        error = self.activation_prime(outputs[-1]) * (correct - outputs[-1])
        self.w[-1] -= self.learning_rate * error * outputs[-2]
        for i in range(len(self.layouts) - 2, 0):
            self.w[-i] -= self.learning_rate * self.activation_prime(outputs[-i]) * outputs[-i - 1]
        pass

    def launch_test(self, test_set):
        total_score = int()
        for record in test_set:
            outputs = list()
            inputs = record.split(',')
            outputs.append(np.asfarray(inputs[1:]) / 255.0 * 0.99 + 0.01)
            correct = int(inputs[0])
            print("Верный ответ: ", correct)
            answer = np.argmax(self.propagation(outputs)[-1])
            print("Полученный ответ: ", answer)
            if answer == correct:
                total_score += 1
        print("Точность: ", str(total_score / len(test_set)))

    def train_without_batch(self, train_set, epoch=1):
        outputs = list()
        for i in range(epoch):
            for record in train_set:
                inputs = record.split(',')
                outputs.append(np.asfarray(inputs[1:]) / 255.0 * 0.99 + 0.01)
                correct = int(inputs[0])
                self.back_propagation(self.propagation(outputs), correct)

