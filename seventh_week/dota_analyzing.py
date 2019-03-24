"""The task of machine learning - check who will win in online game Dota2
================================================

Contents
--------
Script predict a winner.

Functions
-----------
::

 prepare_data_set                   --- Preparation of the train and test data sets
 characters                         --- Counting characters, preparing bag of words and calculating logistic regression
 gradient_boosting                  --- Counting cross validation over gradient boosting
 logistic_regression                --- Counting cross validation over logistic regression
 logistic_regression_with_bag       --- Counting cross validation over logistic regression with using "bag of words"
 final_check                        --- Predicts of radiants wins

"""
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def prepare_data_set():
    data_set_train = pd.read_csv(
        './data/features.csv',
        index_col='match_id'
    )
    data_set_test = pd.read_csv(
        './data/features_test.csv',
        index_col='match_id'
    )
    columns = list()

    columns.append('duration')
    columns.append('tower_status_radiant')
    columns.append('tower_status_dire')
    columns.append('barracks_status_radiant')
    columns.append('barracks_status_dire')

    data_set_train.drop(
        columns=columns,
        axis=1,
        inplace=True
    )

    X_train_dirty = data_set_train[[col for col in data_set_train.columns if col not in ['radiant_win']]]
    X_test_dirty = data_set_train[[col for col in data_set_train.columns if col not in ['radiant_win']]]
    columns.clear()

    for i in range(1, 6):
        columns.append('r{}_hero'.format(i))
        columns.append('d{}_hero'.format(i))

    columns.append('lobby_type')

    data_set_train.drop(
        columns=columns,
        axis=1,
        inplace=True
    )
    data_set_test.drop(
        columns=columns,
        axis=1,
        inplace=True
    )

    X_train = data_set_train[[col for col in data_set_train.columns if col not in ['radiant_win']]]
    y_train = data_set_train['radiant_win']
    X_test = data_set_test[[col for col in data_set_test.columns if col not in ['radiant_win']]]

    return X_train, y_train, X_test, X_train_dirty, X_test_dirty


def characters(X, y):
    characters = pd.read_csv(
        './data/dictionaries/heroes.csv',
        index_col='id'
    )

    print(
        'There are',
        str(len(characters)),
        'characters'
    )

    bag_matrix = np.zeros(
        (
            X.shape[0],
            len(characters)
        )
    )

    for i, match_id in enumerate(X.index):
        for j in range(5):
            bag_matrix[i, X.ix[match_id, 'r%d_hero' % (j + 1)] - 1] = 1
            bag_matrix[i, X.ix[match_id, 'd%d_hero' % (j + 1)] - 1] = -1

    return pd.DataFrame(
        bag_matrix,
        index=X.index
    )


def gradient_boosting(X, y):
    kf = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    scores = []
    trees_count = [10, 20, 30, 50, 100, 250]

    for num in trees_count:
        print('Count of trees - ', str(num))
        start_time = datetime.now()

        clf = GradientBoostingClassifier(
            n_estimators=num,
            random_state=42
        )

        cross_val = ms.cross_val_score(
            clf,
            X,
            y,
            cv=kf,
            scoring='roc_auc',
            n_jobs=-1
        )
        print('Time elapsed:', datetime.now() - start_time)
        print(cross_val)
        scores.append(np.mean(cross_val))

    max_score = max(scores)
    max_score_index = scores.index(max_score)

    return trees_count[max_score_index], max_score


def logistic_regression(X, y):
    scaler = StandardScaler()
    X_temp = scaler.fit_transform(X)

    kf = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    scores = []

    range_of_C = [10.0 ** i for i in range(-2, -1)]

    for C in range_of_C:
        start_time = datetime.now()
        model = LogisticRegression(
            C=C,
            random_state=42,
            n_jobs=-1
        )

        model_scores = cross_val_score(
            model,
            X_temp,
            y,
            cv=kf,
            scoring='roc_auc',
            n_jobs=-1
        )

        print(
            'C =',
            str(C),
            'Time elapsed:',
            datetime.now() - start_time
        )
        print(model_scores)
        scores.append(np.mean(model_scores))

    max_score = max(scores)
    max_score_index = scores.index(max_score)

    return range_of_C[max_score_index], max_score


def logistic_regression_with_bag(X, y, X_characters, X_test):
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), index=X.index)
    X = pd.concat([X, X_characters], axis=1)

    X_test = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index)
    X_test = pd.concat([X_test, X_characters], axis=1)
    C, score = logistic_regression(X, y)

    return C, score, X, X_test


def final_check(C, X_train, y_train, X_test):
    model = LogisticRegression(
        C=C,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_test = model.predict_proba(X_test)[:, 1]

    result = pd.DataFrame({'radiant_win': y_test}, index=X_test.index)
    result.index.name = 'match_id'
    result.to_csv('./data/predict.csv')

    return min(y_test), max(y_test)


if __name__ == '__main__':
    pd.set_option(
        'display.max_columns',
        None
    )
    X_train, y_train, X_test, X_train_dirty, X_test_dirty = prepare_data_set()

    desc = X_train.describe()

    rows_count = len(X_train)
    counts = desc.T['count']
    counts_na = counts[counts < rows_count]

    X_train.fillna(int(0), inplace=True)
    X_test.fillna(int(0), inplace=True)
    X_train_dirty.fillna(int(0), inplace=True)
    X_test_dirty.fillna(int(0), inplace=True)

    trees_number, trees_score = gradient_boosting(X_train, y_train)
    C_dirty, score_dirty = logistic_regression(X_train_dirty, y_train)
    C, score = logistic_regression(X_train, y_train)

    X_characters = characters(X_train_dirty, y_train)

    C_bag, score_bag, X_train_char, X_test_char = logistic_regression_with_bag(
        X_train_dirty,
        y_train,
        X_characters,
        X_test_dirty
    )

    X_test_char.fillna(int(0), inplace=True)

    min_y_predict, max_y_predict = final_check(C_bag, X_train_char, y_train, X_test_char)

    print(
        counts_na.sort_values().apply(lambda c: (rows_count - c) / rows_count)
    )

    print(
        'Trees number =',
        trees_number,
        'score =',
        trees_score,
        sep=' '
    )

    print(
        'Dirty result C =',
        C_dirty,
        'score =',
        score_dirty,
        sep=' '
    )

    print(
        'Pure result C =',
        C,
        'score =',
        score,
        sep=' '
    )

    print(
        'Bag of words result C =',
        C_bag,
        'score =',
        score_bag,
        sep=' '
    )

    print(
        'Minimum of predicts:',
        min_y_predict,
        sep=' '
    )

    print(
        'Maximum of predicts:',
        max_y_predict,
        sep=' '
    )
