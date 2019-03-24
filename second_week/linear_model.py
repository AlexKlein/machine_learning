"""The task of machine learning - normalization of attributes
================================================

Contents
--------
Script shows that Perceptron with normalization of attributes is better.

Task
----
0. Download the training and test data sets.
1. Train Perceptron with standard parameters and random_state = 241.
2. Calculate the quality (proportion of correctly classified objects).
3. Normalize the training and test set using the StandardScaler class.
4. Train Perceptron on the new data set. Find the proportion of correct answers on the test data set.
5. Find the difference between the quality on the test sample after normalization and the quality before it.

"""
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas


if __name__ == '__main__':
    clf = Perceptron(random_state=241)
    data_set_test = pandas.read_csv(
        'perceptron-test.csv',
        index_col=False,
        header=None,
        names=[
            'type',
            'first',
            'second'
        ]
    )
    data_set_train = pandas.read_csv(
        'perceptron-train.csv',
        index_col=False,
        header=None,
        names=[
            'type',
            'first',
            'second'
        ]
    )
    output_file = open('output.txt', 'w', encoding='ANSI')
    scales = StandardScaler()

    X_test = np.array(data_set_test[['first', 'second']])
    y_test = np.array(data_set_test['type'])

    X_train = np.array(data_set_train[['first', 'second']])
    y_train = np.array(data_set_train['type'])

    clf.fit(
        X_train,
        y_train
    )
    predictions_pure = clf.predict(X_test)
    accuracy_pure = accuracy_score(y_test, predictions_pure)

    X_train_scaled = scales.fit_transform(X_train)
    X_test_scaled = scales.transform(X_test)

    clf.fit(
        X_train_scaled,
        y_train
    )
    predictions_normalized = clf.predict(X_test_scaled)

    accuracy_normalized = accuracy_score(y_test, predictions_normalized)

    print(
        round(
            abs(
                accuracy_pure-accuracy_normalized
            ),
            3
        ),
        file=output_file
    )

    output_file.close()
