"""The task of machine learning - preparing data using the Pandas library
================================================

Contents
--------
Script Titanic uses the SciPy and Pandas library and calls 6 functions.

Task
----
0. Download titanic.csv
1. How many men and women rode the ship?
2. What part of the passengers managed to survive?
3. What proportion of first-class passengers were among all the passengers?
4. How old were the passengers? Calculate the average and median age of the passengers.
5. Do the number of siblings / spouses correlate with the number of parents / children?
6. What is the most popular female name on the ship?

Functions
-----------
::

 male_female_count            --- Counting the number of passengers of men and women
 survived_percent             --- Counting the percentage of survivors
 first_class_percent          --- Counting the percentage of first-class passengers
 average_age                  --- Calculation of the average and median age of passengers
 relation_correlation         --- Calculation of the correlation between the passengers parents its siblings / spouses
 popular_female_name          --- Choosing the most popular female name

"""
from scipy import stats as sp
import pandas


def male_female_count():
    print(
        dict(
            data_set['Sex'].value_counts()).get('male'),
        dict(
            data_set['Sex'].value_counts()).get('female'),
        sep=' '
    )


def survived_percent():
    print(
        round(
            100 *
            int(
                dict(
                    data_set['Survived'].value_counts()).get(1)
            ) /
            int(
                data_set['Survived'].count()
            ),
            2
        )
    )


def first_class_percent():
    print(
        round(
            100 *
            int(
                dict(
                    data_set['Pclass'].value_counts()).get(1)
            ) /
            int(
                data_set['Pclass'].count()
            ),
            2
        )
    )


def average_age():
    print(
        round(
            data_set['Age'][pandas.notnull(data_set['Age'])].aggregate(sum) /
            data_set['Age'][pandas.notnull(data_set['Age'])].count(),
            2
        ),
        round(
            sorted(
                list(
                    data_set['Age'][pandas.notnull(data_set['Age'])]
                )
            )
            [int(
                round(
                    data_set['Age'][pandas.notnull(data_set['Age'])].count() /
                    2
                )
            )
            ],
            2
        )
    )


def relation_correlation():
    print(
        round(
            sp.stats.pearsonr(
                list(
                    data_set['SibSp'][pandas.notnull(data_set['SibSp'])]
                ),
                list(
                    data_set['Parch'][pandas.notnull(data_set['Parch'])]
                )
            )[0],
            2
        )
    )


def popular_female_name():
    female_list = list(data_set['Name'][data_set['Sex'] == 'female'])
    final_list = list()
    name_list = list()

    for name in female_list:
        if str(name).find('Miss') > 0:
            name_list.append(
                name[str(name).find(
                        'Miss'
                ) + 6:
                     str(name).find(
                         ' ',
                         str(name).find(
                             'Miss'
                         ) + 7
                     )]
            )
        elif str(name).find('Mrs') > 0:
            name_list.append(
                name[str(name).find('(')+1:
                     str(name).find(' ',
                                    str(name).find('(') + 2)
                ]
            )

    for name in set(name_list):
        final_list.append(
            (
                name_list.count(name),
                name
            )
        )

    print(
        sorted(
            final_list,
            reverse=True)[0][1],
        file=output_file
    )


if __name__ == '__main__':
    data_set = pandas.read_csv(
        'titanic.csv',
        index_col='PassengerId'
    )
    output_file = open(
        'output.txt',
        'w',
        encoding='utf8'
    )

    male_female_count()
    survived_percent()
    first_class_percent()
    average_age()
    relation_correlation()
    popular_female_name()

    output_file.close()
