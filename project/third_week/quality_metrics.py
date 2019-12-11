"""The task of machine learning - quality metrics
================================================

Contents
--------
Script checks qualities of metrics.

Task
----
0. Download the classification.csv file.
1. Fill in the classification error table.
2. Calculate the main quality metrics of the classifier.
3. Download the scores.csv file.
4. Calculate the area under the ROC curve for each classifier. Which classifier has the highest AUC-ROC metric value.
5. Which classifier achieves the greatest precision with recall of at least 70%?

Functions
-----------
::

 class_error            --- Classification error table filling
 metrics_score          --- Calculates metrics
 best_metric            --- Calculates the highest AUC-ROC metric
 max_precision          --- Calculates which classifier achieves the greatest precision

"""
import pandas
import sklearn.metrics as mt


def class_error():
    tn, fp, fn, tp = mt.confusion_matrix(
        data_set_class['true'],
        data_set_class['pred'],
    ).ravel()

    print(
        tn,
        fp,
        fn,
        tp,
        sep=' '
    )


def metrics_score():
    accuracy = mt.accuracy_score(
        data_set_class['true'],
        data_set_class['pred']
    )
    precision = mt.precision_score(
        data_set_class['true'],
        data_set_class['pred']
    )
    recall = mt.recall_score(
        data_set_class['true'],
        data_set_class['pred']
    )
    f_score = mt.f1_score(
        data_set_class['true'],
        data_set_class['pred']
    )

    print(
        accuracy,
        precision,
        recall,
        f_score,
        sep=' '
    )


def best_metric():
    max_score = list()
    max_score.append(
        (
            mt.roc_auc_score(
                data_set_score['true'],
                data_set_score['score_logreg']
            ),
            'score_logreg'
        )
    )
    max_score.append(
        (
            mt.roc_auc_score(
                data_set_score['true'],
                data_set_score['score_svm']
            ),
            'score_svm'
        )
    )
    max_score.append(
        (
            mt.roc_auc_score(
                data_set_score['true'],
                data_set_score['score_knn']
            ),
            'score_knn'
        )
    )
    max_score.append(
        (
            mt.roc_auc_score(
                data_set_score['true'],
                data_set_score['score_tree']
            ),
            'score_tree'
        )
    )

    print(
        sorted(
            max_score,
            reverse=True
        )[0][1],
        sep=' '
    )


def max_precision():
    max_precision = list()

    for metric in [col for col in data_set_score.columns if col not in ['true']]:
        precision, recall, thresholds = mt.precision_recall_curve(
            data_set_score['true'],
            data_set_score[metric]
        )
        temp = precision[(recall >= 0.7)].max()
        max_precision.append(
            (
                temp,
                metric
            )
        )

    print(
        sorted(
            max_precision,
            reverse=True
        )[0][1],
        file=output_file
    )


if __name__ == '__main__':
    data_set_class = pandas.read_csv(
        'classification.csv'
    )
    data_set_score = pandas.read_csv(
        'scores.csv'
    )
    output_file = open(
        'output.txt',
        'w',
        encoding='ANSI')

    class_error()
    metrics_score()
    best_metric()
    max_precision()

    output_file.close()
