from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from itertools import combinations
from combinatorics import C_sum
import numpy as np
FIG_SIZE = (10, 7)


features, target = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.30)

clf = GaussianNB()
clf.fit(X_train, y_train)

pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)

diff_train = pred_train - y_train
diff_test = pred_test - y_test
err_train = np.count_nonzero(diff_train)
err_test = np.count_nonzero(diff_test)

print("Ошибок на обучающей:\t\t",err_train)
print("Ошибок на тестовой:\t\t",err_test)

dim = features.shape[1]
size = [n for n in range(dim)]
amount = C_sum(dim) - 1
errors_train = np.zeros(amount).astype(int)
errors_test = np.zeros(amount).astype(int)
features_list = list()
l = [errors_train, errors_test]
for length in range(1,dim+1):
    for i, subset in enumerate(combinations(size,length)):

        columns = list(subset)
        X_train_part = X_train[:,columns]
        X_test_part = X_test[:,columns]

        clf.fit(X_train_part, y_train)
        pred_train = clf.predict(X_train_part)
        pred_test = clf.predict(X_test_part)
        diff_train = pred_train  - y_train
        diff_test = pred_test  - y_test
        errors_train[i] = np.count_nonzero(diff_train)
        errors_test[i] = np.count_nonzero(diff_test)
        #print(errors_test[i])


