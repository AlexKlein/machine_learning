"""The task of machine learning - text analysis
================================================

Contents
--------
Script loads the data set with 20 groups and selects most important words.

Task
----
0. Download items from the news data set.
1. Calculate TF-IDF signs for all texts.
2. Pick the minimum best C parameter from the set.
3. Train the SVM over the entire data set with the optimal C parameter found in the previous step.
4. Find the 10 words with the highest absolute weight value.

"""
from sklearn import datasets
from sklearn import model_selection as ms
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd


if __name__ == '__main__':
    clf = svm.SVC(
        kernel='linear',
        random_state=241,
    )
    cv = ms.KFold(
        n_splits=5,
        shuffle=True,
        random_state=241
    )
    grid = {
        'C': np.power(
            10.0,
            np.arange(
                -5,
                6
            )
        )
    }
    newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
    )
    output_file = open(
        'output.txt',
        'w',
        encoding='ANSI'
    )
    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target

    gs = ms.GridSearchCV(
        clf,
        grid,
        scoring='accuracy',
        cv=cv
    )
    gs.fit(X, y)
    # doesn't work in 20-th version of scikit-learning
    # for a in gs.grid_scores_:
    #     print(a.mean_validation_score)
    #     print(a.parameters)

    k_score = list()

    for k in range(-5, 6, 1):
        k_score.append(
            (
                (
                    ms.cross_val_score(
                        estimator=svm.SVC(
                            kernel='linear',
                            random_state=241,
                            C=10 ** k
                        ),
                        X=X,
                        y=y,
                        cv=cv,
                        scoring='accuracy'
                    )
                ).mean(),
                k
            )
        )

    best_c = sorted(
        k_score,
        reverse=True
    )[0][1]

    clf = svm.SVC(
        kernel='linear',
        random_state=241,
        C=10 ** best_c
    )

    clf.fit(X, y)
    words_set = pd.DataFrame(clf.coef_.toarray()).transpose()
    top10 = abs(words_set).sort_values([0], ascending=False).head(10)
    indexes = top10.index
    words = list()

    for i in indexes:
        feature_mapping = vectorizer.get_feature_names()
        words.append(feature_mapping[i])

    print(
        ",".join(
            sorted(
                words
            )
        ),
        file=output_file
    )

    output_file.close()
