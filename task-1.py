import numpy as np
from scipy import stats


def confidence_interval_for_a(data, epsilon, sigma):
    mean = np.mean(data)
    size = len(data)
    # quantile = np.quantile(data, 1 - epsilon / 2)
    quantile = stats.norm.ppf(1 - epsilon / 2)
    print(quantile)
    print(sigma)

    return sorted((mean - quantile * sigma / np.sqrt(size), mean + quantile * sigma / np.sqrt(size)))


def confidence_interval_for_sigma(data, epsilon, a=None):
    size = len(data)

    if a:
        s_2_1 = sum([(X_i - a) ** 2 for X_i in data])
        print(s_2_1)
        q1 = stats.chi2.ppf(epsilon / 2, size)
        q2 = stats.chi2.ppf(1 - epsilon / 2, size)

        print(q1, q2)

        return sorted((s_2_1 / q1, s_2_1 / q2))
    else:
        var = variance_unbiased(data)
        q1 = stats.chi2.ppf(epsilon / 2, size - 1)
        q2 = stats.chi2.ppf(1 - epsilon / 2, size - 1)

        print(q1, q2)
        print(var)

        return sorted(((size - 1) * var / q2, (size - 1) * var / q1))


def variance_unbiased(data):
    size = len(data)
    return size / (size - 1) * variance(data)


def variance(data, mean=None):
    mean = np.mean(data) if not mean else mean
    return sum([data[i] ** 2 for i in range(len(data))]) / len(data) - mean ** 2


# check this!
def F_test(X, Y, epsilon):
    var_X, var_Y = variance_unbiased(X), variance_unbiased(Y)

    if var_X > var_Y:
        F = var_X / var_Y
        q = stats.f.ppf(1 - epsilon, len(X), len(Y))
    else:
        F = var_Y / var_X
        q = stats.f.ppf(1 - epsilon, len(Y), len(X))

    print('F test', F)
    print(q)
    print(var_X)
    print(var_Y)
    return F < q


def student_test(X, Y, epsilon):
    X_mean, Y_mean = np.mean(X), np.mean(Y)
    X_len, Y_len = len(X), len(Y)
    X_var, Y_var = variance(X), variance(Y)

    t = ((X_mean - Y_mean) / np.sqrt(X_var * X_len + Y_var * Y_len)) * np.sqrt(
        (X_len + Y_len - 2) / (1 / X_len + 1 / Y_len))
    print('student test', t)
    q = stats.t.ppf(1-epsilon/2, X_len+Y_len-2)
    print(q)
    print(X_mean)
    print(Y_mean)

    return np.abs(t) < q


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        ls = []
        for line in file.readlines():
            ls.extend([float(x) for x in line.split()])

        return np.array(ls)


data = read_data('data_1_tmp.csv')
size = len(data)
epsilon = 0.05
sigma = np.sqrt(0.5)
a = -1

# print(confidence_interval_for_a(data, epsilon, sigma))
# print(variance_unbiased(data))
# print(confidence_interval_for_a(data, epsilon, np.sqrt(variance_unbiased(data))))
# print(confidence_interval_for_sigma(data, epsilon, a))
# print(confidence_interval_for_sigma(data, epsilon))

X = data[:20]
Y = data[20:]
# print(F_test(X, Y, epsilon))

print(student_test(data[:20], data[20:], epsilon))
