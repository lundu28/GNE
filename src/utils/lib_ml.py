from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as CART
from sklearn.utils import shuffle
import numpy as np
import sys
import re

class MachineLearningLib(object):
    @staticmethod
    def svm(X, y):
        clf = svm.SVC()
        clf.fit(X, y)
        return clf

    @staticmethod
    def infer(clf, X, y_true = None):
        y = clf.predict(X)
        if y_true is None:
            return y
        else:
            return y, clf.score(X, y_true)

    @staticmethod
    def logistic(X, y):
        clf = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', max_iter = 10000)
        clf.fit(X, y)
        return clf

    @staticmethod
    def cart(X, y):
        clf = CART()
        clf.fit(X, y)
        return clf

