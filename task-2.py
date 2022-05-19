from math import inf
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from statsmodels.distributions.empirical_distribution import ECDF


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        ls = []
        for line in file.readlines():
            ls.extend([float(x) for x in line.split()])

        return np.array(ls)


data = read_data('data_2_tmp.csv')
n = len(data)


def empiric_function(t):
    return sum([1 if x_i < t else 0 for x_i in data]) / n


def plot_func(f):
    plt.style.use('_mpl-gallery')

    x = np.linspace(0, 1, 1000)
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
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    for x_, y_ in zip(segments_x, segments_y):
        ax.plot(x_, y_, color='r', zorder=0)

    ax.set_xlabel('t')
    ax.set_ylabel('F(t)')
    ax.set_title("Empiric Function Plot")

    ax.plot(x, x, color='b', zorder=0)

    # ax.annotate('sup point',
    #         xy=(0.349, 0.35),
    #         xytext=(0.4, 0.25),
    #         arrowprops = dict(facecolor='black', shrink=0.05))

    plt.show()


def plot_hist(f):
    plt.style.use('_mpl-gallery')

    x = np.linspace(0, 1, 1001)
    y = np.array([f(t) for t in x])

    fig = plt.figure(figsize=(20, 10), layout='constrained', dpi=200)
    ax = fig.add_subplot()

    ax.hist(y, bins=30)

    ax.set_xlabel('t')
    ax.set_ylabel('Frequency')
    ax.set_title("Empiric Function Hist")

    plt.show()


def k_test(X, epsilon=None):
    X_sorted = np.sort(X)
    N = len(X)

    point = ""
    sup = 0

    for i in range(1, N+1):
        if np.abs(X_sorted[i - 1] - (i-1)/N) > sup:
            sup = np.abs(X_sorted[i - 1] - (i-1)/N)
            point = f"X({i}) = {X_sorted[i - 1]}"

        if np.abs(X_sorted[i - 1] - i/N) > sup:
            sup = np.abs(X_sorted[i - 1] - i/N)
            point = f"X({i}) = {X_sorted[i - 1]}"

    print(sup)
    print(point)

    K = np.sqrt(N) * sup
    q = stats.ksone.ppf(1-epsilon, N)

    print('K test', K)
    print(q)

    print('p-value', stats.kstest(X, 'uniform')[1])

    return K < q


def chi_square(X):
    return stats.chisquare(X)


epsilon = 0.05
# plot_func(ECDF(data))
# plot_hist(ECDF(data))
# print(k_test(data, epsilon))
print(chi_square(data))
