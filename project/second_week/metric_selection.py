"""The task of machine learning - selection of the optimal metric
================================================

Contents
--------
Script selects the optimal parameter.

Task
----
0. Load a Boston data set.
1. Scale attributes.
2. Enumerate different variants of the parameter metrics p on the grid from 1 to 10.
3. Determine at which p the quality on cross-validation turned out to be optimal.

"""
from sklearn import datasets as dt
from sklearn import model_selection as ms
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import sklearn


if __name__ == '__main__':
    data_set = dt.load_boston()
    output_file = open('output.txt', 'w', encoding='ANSI')
    X = data_set['data']
    y = data_set['target']
    X = sklearn.preprocessing.scale(X)

    kf = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    k_score = list()

    for i in np.linspace(1, 10, num=200):
        knr = KNeighborsRegressor(
            p=i,
            n_neighbors=5,
            weights='distance',
            metric='minkowski'
        )
        k_score.append(
            (
                (
                    ms.cross_val_score(
                        estimator=knr,
                        X=X,
                        y=y,
                        cv=kf,
                        scoring='neg_mean_squared_error'
                    )
                ).mean(),
                i
            )
        )

    print(
        round(
            sorted(
                k_score,
                reverse=True
            )[0][1],
            2
        ),
        file=output_file
    )

    output_file.close()
