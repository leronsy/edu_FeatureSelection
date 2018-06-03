import itertools as itt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine, load_boston
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from combinatorics import combinations_sum


def classification_errors_counter(predicted, known):
    diff = predicted - known
    errors = np.count_nonzero(diff)
    return errors


def compare_regression_mse(current, correct):
    mse = mean_squared_error(current, correct)
    return mse


def printlist(list_for_print):
    print("-" * 20)
    for item in list_for_print:
        print(item)
    print("-" * 20)


def main(dataset, classifier, error_function):
    features, target = dataset(return_X_y=True)
    clf = classifier
    err_fun = error_function

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30)

    dimension = features.shape[1]
    combinations_amount = combinations_sum(dimension) - 1
    combination_length_lst = [n for n in range(dimension)]

    errors_train = np.zeros(combinations_amount)
    errors_test = np.zeros(combinations_amount)

    print('Перебор из {0} наборов признаков.'.format(combinations_amount))

    plt.figure(figsize=(6, 7), dpi=200)
    min_by_len = np.zeros((2, dimension))
    features_lst = list()
    i = 0
    for length in range(1, dimension + 1):
        i_prev = i
        for subset in itt.combinations(combination_length_lst, length):
            columns = list(subset)
            features_lst.append(columns)

            x_train_part = x_train[:, columns]
            x_test_part = x_test[:, columns]

            clf.fit(x_train_part, y_train)

            predict_train = clf.predict(x_train_part)
            predict_test = clf.predict(x_test_part)

            errors_train[i] = err_fun(predict_train, y_train)
            errors_test[i] = err_fun(predict_test, y_test)

            i += 1
        plt.scatter([length for _ in range(i_prev, i)], errors_test[i_prev:i], s=20)
        min_by_len[0, length - 1] = length
        min_by_len[1, length - 1] = errors_test[i_prev:i].min()

    plt.axhline(errors_test.min(), ls='dashed', c='red')
    plt.plot(min_by_len[0], min_by_len[1], c='grey')
    plt.axis([0, dimension + 1, errors_test.min() - 1, errors_test.max() + 1])

    minimums_test = np.where(errors_test == errors_test.min())
    best_col_set_test = features_lst[minimums_test[0][0]]
    columns_minimal_number = len(best_col_set_test)
    col_lst = []
    for i in minimums_test[0]:
        lst = features_lst[i]
        if len(lst) == columns_minimal_number:
            col_lst.append(lst)

    plt.legend(
        ('min = ' + str("{0:3.2f} on {1:2d} p").format(errors_test.min(), len(best_col_set_test)),))
    plt.show()

    print("Лучшие минимальные наборы по тестам:\t")
    printlist(col_lst)
    print("Ошибки у наборов\t train | test")
    print('{0:17s}\t{1:6.2f} |{2:5.2f}'.format("Лучшие", errors_train.min(), errors_test.min()))
    print('{0:17s}\t{1:6.2f} |{2:5.2f}'.format("Полный", errors_train[-1], errors_test[-1]))


if __name__ == '__main__':
    main(load_wine, GaussianNB(), classification_errors_counter)
    # main(load_boston, LinearRegression(), compare_regression_mse)
