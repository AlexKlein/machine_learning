"""The task of machine learning - select the number of neighbors
================================================

Contents
--------
Script nearest_neighbors selects the optimal number of neighbors.

Task
----
0. Download the Wine data set.
1. Extract the attributes and classes from the data.
2. Quality assessment should be carried out by the method of cross-validation of 5 blocks.
3. Find the accuracy of classification for cross-validation for the method of k nearest neighbors.
4. Scale attributes, find the optimal k on cross-validation.
5. What value of k turned out to be optimal after reducing the signs to one scale?

"""
from sklearn import model_selection as ms
from sklearn.neighbors import KNeighborsClassifier
import pandas
import sklearn


if __name__ == '__main__':
    data_set = pandas.read_csv('wine.data', index_col=False, header=None)
    output_file = open('output.txt', 'w', encoding='ANSI')
    data_set.set_axis(
        axis=1,
        labels=['type',
                'alcohol',
                'malic_acid',
                'ash',
                'alcalinity_of_ash',
                'magnesium',
                'total_phenols',
                'flavanoids',
                'nonflavanoid_phenols',
                'proanthocyanins',
                'color_intensity',
                'hue',
                'of_diluted_wines',
                'proline'])
    
    attributes_cols = [col for col in data_set.columns if col not in ['type']]
    X = data_set[attributes_cols]
    y = data_set['type']

    kf = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    k_score = list()

    for k in range(1, 51):
        k_score.append(
            (
                (
                    ms.cross_val_score(
                        estimator=KNeighborsClassifier(
                            n_neighbors=k
                        ),
                        X=X,
                        y=y,
                        cv=kf,
                        scoring='accuracy'
                    )
                ).mean(),
                k
            )
        )

    print(
        sorted(
            k_score,
            reverse=True
        )[0][1]
    )

    print(
        round(
            sorted(
                k_score,
                reverse=True
            )
            [0][0],
            2
        )
    )

    X = sklearn.preprocessing.scale(X)
    k_score.clear()

    for k in range(1, 51):
        k_score.append(
            (
                (
                    ms.cross_val_score(
                        estimator=KNeighborsClassifier(
                            n_neighbors=k
                        ),
                        X=X,
                        y=y,
                        cv=kf,
                        scoring='accuracy'
                    )
                ).mean(),
                k
            )
        )

    print(
        sorted(
            k_score,
            reverse=True
        )[0][1]
    )

    print(
        round(
            sorted(
                k_score,
                reverse=True
            )
            [0][0],
            2
        ),
        file=output_file
    )
    output_file.close()
