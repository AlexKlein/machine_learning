"""The task of machine learning - logistic regression
================================================

Contents
--------
Script calculates auc-roc scores.

Task
----
0. Download a data set from the data-logistic.csv file.
1. Make sure that the correct formulas for gradient descent are written out above.
2. Implement a gradient descent for normal and L2-regularized logistic regression.
3. Run the gradient descent and bring it to convergence.
4. What is the significance of AUC-ROC in training without regularization and in its use?
5. Try changing the step length. Will the algorithm converge if you take longer steps?
6. Try changing the initial approximation. Does it affect anything?

Functions
-----------
::

 sigmoid                      --- Monotone non-decreasing function with a range of values [0, 1]
 distance                     --- Calculates the distance between two vectors
 logistic_regression          --- Gradient descent

"""
import numpy as np
import pandas
import sklearn.metrics as mt


def sigmoid(x):
    return 1.0 / 1 + np.exp(-x)


def distance(a, b):
    return np.sqrt(
        np.square(
            a[0] - b[0]
        ) +
        np.square(
            a[1] - b[1]
        )
    )


def logistic_regression(X, y, C):
    epsilon = 1e-5
    k = 0.1
    max_iter = 10000
    w = [0.0, 0.0]
    w1, w2 = w
    for i in range(max_iter):

        w1_new = w1 + k * np.mean(
            y * X[:, 0] * (
                    1 - (
                        1.0 / (
                            1 + np.exp(
                                -y * (
                                        w1 * X[:, 0] + w2 * X[:, 1]
                                )
                            )
                        )
                    )
            )
        ) - k * C * w1
        w2_new = w2 + k * np.mean(
            y * X[:, 1] * (
                    1 - (
                        1.0 / (
                            1 + np.exp(
                                -y * (
                                        w1 * X[:, 0] + w2 * X[:, 1]
                                )
                            )
                        )
                    )
            )
        ) - k * C * w2

        if distance((w1_new, w2_new), (w1, w2)) < epsilon:
            break

        w1, w2 = w1_new, w2_new

    predictions = list()

    for j in range(len(X)):
        t1 = -w1 * X[j, 0] - w2 * X[j, 1]
        s = sigmoid(t1)
        predictions.append(s)

    return predictions


if __name__ == '__main__':
    data_set = pandas.read_csv(
        'data-logistic.csv',
        index_col=False,
        header=None,
        names=[
            'type',
            'first',
            'second'
        ]
    )
    output_file = open('output.txt', 'w', encoding='ANSI')

    X = np.array(data_set[['first', 'second']])
    y = np.array(data_set['type'])

    p0 = logistic_regression(X, y, 0)
    p1 = logistic_regression(X, y, 10)

    print(
        round(
            mt.roc_auc_score(
                y,
                p0
            ),
            3
        ),
        round(
            mt.roc_auc_score(
                y,
                p1
            ),
            3
        ),
        sep=' ',
        file=output_file
     )

    output_file.close()
