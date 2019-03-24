"""The task of machine learning - linear regression
================================================

Contents
--------
Script makes salary forecast by job description.

Task
----
0. Download data about job descriptions and corresponding annual salaries from the salary-train.csv file.
1. Perform pre-processing:
1.1. Bring the text to lower case.
1.2. Replace everything except letters and numbers with spaces.
1.3. Apply TfidfVectorizer to convert texts. Leave only those words that are found in at least 5 objects.
1.4. Replace gaps in the LocationNormalized and ContractTime columns with the special string 'nan'.
1.5. Use the DictVectorizer to obtain one-hot-coding of LocationNormalized and ContractTime.
1.6. Combine all the obtained signs into a single "objects-attributes" matrix.
2. Train the ridge regression with the parameters alpha = 1 and random_state = 241.
3. Build predictions for the two examples from the salary-test-mini.csv file.

Functions
-----------
::

 prepare_train_data_set          --- Preparation of the initial train data
 prepare_test_data_set           --- Preparation of the initial test data

"""
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd


def prepare_train_data_set():
    data_set_train = pd.read_csv(
        'salary-train.csv',
        index_col=False,
        nrows=50000
    ).applymap(
        lambda x: x.lower() if type(x) == str else x
    )

    for column in [col for col in data_set_train.columns]:
        data_set_train[column] = data_set_train[column].replace(
            '[^a-zA-Z0-9]',
            ' ',
            regex=True
        )

    data_set_train['LocationNormalized'].fillna(
        'nan',
        inplace=True
    )
    data_set_train['ContractTime'].fillna(
        'nan',
        inplace=True
    )

    attributes_cols = [col for col in data_set_train.columns if col not in ['SalaryNormalized']]
    X = data_set_train[attributes_cols]
    y = data_set_train['SalaryNormalized'].astype(np.float)

    return X, y


def prepare_test_data_set():
    data_set_test = pd.read_csv(
        'salary-test-mini.csv',
        index_col=False
    ).applymap(
        lambda x: x.lower() if type(x) == str else x
    )

    for column in [col for col in data_set_test.columns]:
        data_set_test[column] = data_set_test[column].replace(
            '[^a-zA-Z0-9]',
            ' ',
            regex=True
        )

    data_set_test['LocationNormalized'].fillna(
        'nan',
        inplace=True
    )
    data_set_test['ContractTime'].fillna(
        'nan',
        inplace=True
    )

    attributes_cols = [col for col in data_set_test.columns if col not in ['SalaryNormalized']]
    X = data_set_test[attributes_cols]

    return X


if __name__ == '__main__':
    enc = DictVectorizer()
    output_file = open(
        'output.txt',
        'w',
        encoding='ANSI'
    )

    X_train, y_train = prepare_train_data_set()
    X_test = prepare_test_data_set()

    vectorizer = TfidfVectorizer(min_df=5)

    X_train_vector = coo_matrix(
        vectorizer.fit_transform(
            X_train.FullDescription
        )
    )

    X_test_vector = coo_matrix(
        vectorizer.transform(
            X_test.FullDescription
        )
    )

    X_train_category = coo_matrix(
        enc.fit_transform(
            X_train[
                [
                    'LocationNormalized',
                    'ContractTime'
                ]
            ].to_dict(
                'records'
            )
        )
    )

    X_test_category = coo_matrix(
        enc.transform(
            X_test[
                [
                    'LocationNormalized',
                    'ContractTime'
                ]
            ].to_dict(
                'records'
            )
        )
    )

    X_train_stack = hstack([X_train_vector, X_train_category])
    X_test_stack = hstack([X_test_vector, X_test_category])

    clf = Ridge(alpha=1)
    clf.fit(X_train_stack, y_train)

    a, b = clf.predict(X_test_stack)
    print(
        round(
            a,
            2
        ),
        round(
            b,
            2
        ),
        sep=' ',
        file=output_file
    )

    output_file.close()
