from itertools import combinations

from numpy import *
# import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from combinatorics import C_sum

FIG_SIZE = (10, 7)


def printlist(list_for_print):
    for item in list_for_print:
        print(item)


features, target = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30)

clf = GaussianNB()
# clf.fit(X_train, y_train)
#
# predict_train = clf.predict(X_train)
# predict_test = clf.predict(X_test)
#
# diff_train = predict_train - y_train
# diff_test = predict_test - y_test
# err_train = count_nonzero(diff_train)
# err_test = count_nonzero(diff_test)

# print("Ошибок на обучающей:\t\t", err_train)
# print("Ошибок на тестовой: \t\t", err_test)

dim = features.shape[1]
size = [n for n in range(dim)]
amount = C_sum(dim) - 1
errors_train = zeros(amount).astype(int)
errors_test = zeros(amount).astype(int)
features_list = list()
i = 0
for length in range(1, dim + 1):
    for subset in combinations(size, length):
        columns = list(subset)
        features_list.append(columns)
        X_train_part = X_train[:, columns]
        X_test_part = X_test[:, columns]

        clf.fit(X_train_part, y_train)
        predict_train = clf.predict(X_train_part)
        predict_test = clf.predict(X_test_part)
        diff_train = predict_train - y_train
        diff_test = predict_test - y_test
        errors_train[i] = count_nonzero(diff_train)
        errors_test[i] = count_nonzero(diff_test)
        i += 1
# np.set_printoptions(threshold=np.nan)

minimums_train = where(errors_train == errors_train.min())
minimums_test = where(errors_test == errors_test.min())

best_col_set_test = features_list[minimums_test[0][0]]
columns_minimal_number = len(best_col_set_test)
col_list = []

intersection = intersect1d(minimums_train, minimums_test)
print("Пересечение", intersection)
for i in minimums_test[0]:
    lst = features_list[i]
    if len(lst) == columns_minimal_number:
        if intersection.size:
            if i in intersection:
                col_list.append(("+", lst))
            else:
                col_list.append(("-", lst))
        else:
            col_list.append(("-", lst))

print("Лучшие наборы по тестам:\t")
printlist(col_list)
print("Ошибок на обучающей | тестовой у лучших наборов:\t", errors_train.min(), '|', errors_test.min())

print("Ошибок на обучающей | тестовой у полного набора:\t", errors_train[-1], '|', errors_test[-1])

# intersection = intersect1d(minimums_train, minimums_test)
# if intersection.size:
#     print("Пересечение", intersection)
#     best_column_set_mixed = features_list[intersection[0]]
#     print("Лучший набор по тесту и обучению:\t", best_column_set_mixed)
