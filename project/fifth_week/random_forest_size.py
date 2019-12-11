"""The task of machine learning - random forest size
================================================

Contents
--------
The script calculates the minimum number of trees for cross validation > 0.52.

Task
----
0. Download the data from the abalone.csv file. This is a dataset in which you want to predict the age of the shell.
1. Convert the sign Sex into numeric: the value of F should go to -1, I - to 0, M - to 1.
2. Divide the contents of the files into attributes and the target variable.
3. Train a random forest with a different number of trees: from 1 to 50.
4. Determine at what minimum number of trees a random forest shows quality on cross-validation above 0.52.
5. Note the change in quality as the number of trees grows. Is it getting worse?

Functions
-----------
::

 prepare_data_set          --- Preparation of the initial data set

"""
import numpy as np
import pandas as pd
from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def prepare_data_set():
    data_set = pd.read_csv(
        'abalone.csv',
        index_col=False
    )
    data_set['Sex'].replace(
        ['F', 'I', 'M'],
        [-1, 0, 1],
        inplace=True
    )

    return np.array(
        data_set[
            [
                col for col in data_set.columns if col not in ['Rings']
            ]
        ]
    ), np.array(
           data_set['Rings']
       )


if __name__ == '__main__':
    output_file = open(
        'output.txt',
        'w',
        encoding='ANSI'
    )
    clf = RandomForestRegressor(
        n_estimators=100,
        random_state=1
    )
    kf = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=1
    )
    X, y = prepare_data_set()

    for i in range(1, 51):
        clf = RandomForestRegressor(
            n_estimators=i,
            random_state=1
        )

        clf.fit(X, y)
        predictions = clf.predict(X)
        r2 = r2_score(y, predictions)

        score = ms.cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            cv=kf,
            scoring='r2'
        ).mean()

        if round(score, 2) > 0.52:
            print(
                i,
                file=output_file
            )
            break

    output_file.close()
