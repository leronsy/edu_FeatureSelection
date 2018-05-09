import itertools as itt

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from combinatorics import combinations_sum


def printlist(list_for_print):
    print("-" * 20)
    for item in list_for_print:
        print(item)
    print("-" * 20)


features, target = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30)

clf = GaussianNB()

dim = features.shape[1]
size = [n for n in range(dim)]
amount = combinations_sum(dim) - 1

errors_train = np.zeros(amount).astype(np.int)
errors_test = np.zeros(amount).astype(np.int)
features_list = list()

i = 0
for length in range(1, dim + 1):
    for subset in itt.combinations(size, length):
        columns = list(subset)
        features_list.append(columns)
        X_train_part = X_train[:, columns]
        X_test_part = X_test[:, columns]

        clf.fit(X_train_part, y_train)
        predict_train = clf.predict(X_train_part)
        predict_test = clf.predict(X_test_part)
        diff_train = predict_train - y_train
        diff_test = predict_test - y_test
        errors_train[i] = np.count_nonzero(diff_train)
        errors_test[i] = np.count_nonzero(diff_test)
        i += 1

minimums_train = np.where(errors_train == errors_train.min())
minimums_test = np.where(errors_test == errors_test.min())

best_col_set_test = features_list[minimums_test[0][0]]
columns_minimal_number = len(best_col_set_test)
col_list = []

intersection = np.intersect1d(minimums_train, minimums_test)
# print("Пересечение", intersection)
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

# np.set_printoptions(threshold=np.nan)
