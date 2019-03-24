"""The task of machine learning - dimension reduction
================================================

Contents
--------
Script uses principal component analysis with stock indexes.

Task
----
0. Download a data set from the close_prices.csv.
1. On the downloaded data set, train the PCA conversion with the number of components equal to 10.
2. Apply the constructed transformation to the source data and take the values ​​of the first component.
3. Download the Dow Jones index information from the djia_index.csv file.
4. Which company has the most weight in the first component?

Functions
-----------
::

 prices_data_set          --- Preparation of the initial data set with prices
 dow_data_set             --- Preparation of the initial data set with Dow Jones index

"""
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def prices_data_set():
    data_set = pd.read_csv(
        'close_prices.csv',
        index_col=False
    )

    return data_set[
        [
            col for col in data_set.columns if col not in ['date']
        ]
    ]


def dow_data_set():
    data_set = pd.read_csv(
        'djia_index.csv',
        index_col=False
    )

    return data_set[
        [
            col for col in data_set.columns if col not in ['date']
        ]
    ]


if __name__ == '__main__':
    output_file = open('output.txt', 'w', encoding='ANSI')
    iter_sum = 0
    X_dow = dow_data_set()
    X_prices = prices_data_set()

    pca = PCA(n_components=10)
    pca.fit(X_prices)

    for i in range(len(pca.explained_variance_ratio_)):
        iter_sum += pca.explained_variance_ratio_[i]
        if iter_sum >= 0.9:
            print(i + 1)
            break

    price_first_comp = pca.transform(X_prices)
    price_first_comp = price_first_comp[:, 0]
    dow = pca.transform(X_dow)[:, 0]
    print(
        round(
            np.corrcoef(
                price_first_comp,
                dow
            )[0][1],
            2
        )
    )

    print(
        X_prices.columns[
            np.argmax(
                pca.components_[0]
            )
        ],
        file=output_file
    )

    output_file.close()
