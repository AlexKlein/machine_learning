"""The task of machine learning - supporting objects
================================================

Contents
--------
Script selects the supporting objects from the data set.

Task
----
0. Download a data set from the svm-data.csv file.
1. Train a classifier with a linear core, the parameter C = 100000 and random_state = 241.
2. Find the numbers of objects that are supporting (numbering from one).

"""
from sklearn.svm import SVC
import numpy as np
import pandas


if __name__ == '__main__':
    clf = SVC(
        C=100000,
        kernel='linear',
        random_state=241
    )
    data_set_test = pandas.read_csv(
        'svm-data.csv',
        index_col=False,
        header=None,
        names=[
            'type',
            'first',
            'second'
        ]
    )
    output_file = open('output.txt', 'w', encoding='ANSI')

    X = np.array(data_set_test[['first', 'second']])
    y = np.array(data_set_test['type'])

    clf.fit(X, y)

    print(
        *map(
            (lambda x: x + 1),
            list(
                clf.support_
            )
        ),
        sep=',',
        file=output_file
    )

    output_file.close()
