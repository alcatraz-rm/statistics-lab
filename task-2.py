from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return np.array([float(row) for row in file.readlines()])


data = read_data('data_2.csv')
n = len(data)


def empiric_function(t):
    return sum([1 if x_i < t else 0 for x_i in data]) / n


def plot(f):
    plt.style.use('_mpl-gallery')

    x = np.linspace(0, 1, 1001)
    y = np.array([f(t) for t in x])

    x_1, y_1 = [0], [0]
    segments_x, segments_y = [], []

    prev = 0
    for i in range(1, len(x)):
        if y[i] != y[i-1]:
            segments_x.append(x[prev:i])
            segments_y.append(y[prev:i])

            x_1.append(x[i])
            y_1.append(y[i])

            prev = i

    x_1 = np.array(x_1)
    y_1 = np.array(y_1)

    fig = plt.figure(figsize=(20, 10), layout='constrained', dpi=200)
    ax = fig.add_subplot()
    ax.scatter(x_1, y_1, edgecolor='b', zorder=1, marker='<')

    for x_, y_ in zip(segments_x, segments_y):
        ax.plot(x_, y_, color='r', zorder=0)

    # ax.plot(x, y, color='r', zorder=0)
    ax.set_xlabel('t')
    ax.set_ylabel('F(t)')
    ax.set_title("Empiric Function Plot")

    plt.show()


plot(empiric_function)
