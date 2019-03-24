"""The task of machine learning - gradient boosting
================================================

Contents
--------
Script checks the reaction of the molecule.

Task
----
0. Load a sample from the gbm-data.csv and convert it to the numpy array (values ​​parameter in the data frame).
1. Train a GradientBoostingClassifier and for each value from the list [1, 0.5, 0.3, 0.2, 0.1] do the following.
1.1. Use the staged_decision_function method to predict quality in a training and test sample at each iteration.
1.2. Transform the prediction using the sigmoid function using the formula 1 / (1 + e ^ {- y_pred}).
1.3. Calculate and plot the log-loss values.
2. How can you characterize the quality graph on a test sample?
3. Give the minimum value of log-loss on the test sample and the number of iteration at which it is reached.
4. On the same data, train the RandomForestClassifier with the number of trees equal to the number of iterations.

Functions
-----------
::

 prepare_data_set          --- Preparation of the initial data set and converting it in array
 split_data                --- Splitting data set fot train and test sub data sets
 sigmoid                   --- Calculating sigmoid function
 log_loss_res              --- Calculating logistic loss function
 print_graph               --- Printing test_loss / train_loss graphics and calculating min value and its index

"""
import math
import numpy as np
import pandas as pd
from sklearn import metrics as mt
from sklearn import model_selection as ms
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt


def prepare_data_set():
    data_set = pd.read_csv(
            'gbm-data.csv',
            index_col=False
        )

    return np.array(
        data_set[
            [
                col for col in data_set.columns if col not in ['Activity']
            ]
        ]
    ), np.array(
           data_set['Activity']
       )


def split_data(X, y):

    return ms.train_test_split(
        X,
        y,
        test_size=0.8,
        random_state=241
    )


def sigmoid(y_pred):
    return 1.0 / (1.0 + math.exp(-y_pred))


def log_loss_res(clf, X, y):
    res = list()

    for predict in clf.staged_decision_function(X):
        res.append(
            mt.log_loss(
                y,
                [sigmoid(y_pred) for y_pred in predict]
            )
        )

    return res


def print_graph(name, train_loss, test_loss):
    plt.figure()
    plt.plot(
        test_loss,
        'r',
        linewidth=2
    )
    plt.plot(
        train_loss,
        'g',
        linewidth=2
    )
    plt.legend(['test', 'train'])
    # plt.show()
    plt.savefig('rate ' + name + ' figure.png')

    min_metric = min(test_loss)
    index_min_metric = test_loss.index(min_metric)

    return round(
        min_metric,
        2
    ), index_min_metric


if __name__ == '__main__':
    output_file = open('output.txt', 'w', encoding='ANSI')

    X, y = prepare_data_set()
    X_train, X_test, y_train, y_test = split_data(X, y)

    min_loss_results = {}
    for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
        clf = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=250,
            verbose=True,
            random_state=241
        )
        clf.fit(X_train, y_train)

        train_loss = log_loss_res(
            clf,
            X_train,
            y_train
        )
        test_loss = log_loss_res(
            clf,
            X_test,
            y_test
        )

        min_loss_results[learning_rate] = print_graph(
            str(learning_rate),
            train_loss,
            test_loss
        )

    print(
        *min_loss_results[0.2],
        sep=' '
    )

    model = RandomForestClassifier(
        n_estimators=min_loss_results[0.2][1],
        random_state=241
    )
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    test_loss = mt.log_loss(y_test, y_pred)

    print(
        test_loss,
        file=output_file
    )

    output_file.close()
