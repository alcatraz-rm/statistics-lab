import numpy as np
from scipy import stats


def confidence_interval_for_a(data, epsilon, sigma):
    mean = np.mean(data)
    size = len(data)
    quantile = np.quantile(data, 1 - epsilon / 2)

    return sorted((mean - quantile * sigma / np.sqrt(size), mean + quantile * sigma / np.sqrt(size)))


def confidence_interval_for_sigma(data, epsilon, a=None):
    if not a:
        a = np.mean(data)
    size = len(data)

    var = variance(data, a)

    return sorted((var/stats.chi2.ppf(epsilon / 2, size), var/stats.chi2.ppf(1 - epsilon / 2, size)))


def variance_unbiased(data):
    size = len(data)
    return size / (size - 1) * np.var(data)


def variance(data, mean=None):
    mean = np.mean(data) if not mean else mean
    return np.abs(sum([(data[i] - mean) ** 2 for i in range(len(data))]))


def F_test(data_1, data_2, epsilon):
    F = np.var(data_1) / np.var(data_2)

    print(stats.f.cdf(F, len(data_1), len(data_2)))
    return stats.f.cdf(F, len(data_1), len(data_2)) <= epsilon


def student_test(data_1, data_2, epsilon):
    m = len(data_1)
    n = len(data_2)

    # z = (data_1.mean() - data_2.mean()) / np.sqrt()


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return np.array([float(row) for row in file.readlines()])


data = read_data('data_1.csv')
size = len(data)
epsilon = 0.05
sigma = np.sqrt(0.5)
a = -1

print(confidence_interval_for_a(data, epsilon, sigma))
print(confidence_interval_for_a(data, epsilon, variance_unbiased(data)))
print(confidence_interval_for_sigma(data, epsilon, a))
print(confidence_interval_for_sigma(data, epsilon))

print(np.var(data[:20]))
print(np.var(data[20:]))
print(F_test(data[:20], data[20:], epsilon))
