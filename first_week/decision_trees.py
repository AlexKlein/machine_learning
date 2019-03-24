"""The task of machine learning - importance of attributes
================================================

Contents
--------
The Decision Trees script defines the two most important signs of passengers for their survival during a crash.

Task
----
0. Download a data set from the titanic.csv file using the Pandas package.
1. Leave the four characteristics in the data set: passenger's class, ticket's price, age and gender .
2. Notice that the Sex attribute has string values.
3. Select the target variable - it is recorded in the Survived column.
4. There are missing values ​​in the data - for example, for some passengers their age is unknown.
5. Train the decision tree with the parameter random_state = 241 and the rest of the default parameters.
6. Calculate the importance of signs and find the two signs with the greatest importance.

"""
from sklearn.tree import DecisionTreeClassifier as dtc
import numpy as np
import pandas as pd


if __name__ == '__main__':
    answer = ''
    answer_set = set()
    clf = dtc()
    data_set = pd.read_csv('titanic.csv', index_col='PassengerId')
    output_file = open('output.txt', 'w', encoding='ANSI')

    data_set['Sex'].replace(['male', 'female'], [1, 0], inplace=True)

    X = np.array(data_set[['Pclass', 'Fare', 'Age', 'Sex']][pd.notnull(data_set['Age'])])
    y = np.array(data_set[['Survived']][pd.notnull(data_set['Age'])])

    clf.fit(X, y)

    for i in range(4):
        answer_set.add(
            (clf.feature_importances_[i],
             ['Pclass', 'Fare', 'Age', 'Sex'][i]
             )
        )

    answer_set = sorted(answer_set, reverse=True)

    for j in range(2):
        a, b = answer_set[j]
        answer += ',' + b

    print(
        answer[1:],
        file=output_file
    )
