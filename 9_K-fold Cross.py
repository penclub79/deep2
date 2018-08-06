import pandas
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


def import_data():
    #import total dataset
    data = pandas.read_csv('iris_data.csv')

    headers = list(data.columns.values)

    x = data[headers[:-1]]
    y = data[headers[-1:]].values.ravel()

    return x,y

if __name__ == '__main__':
    x, y = import_data()

    skf = StratifiedKFold(n_splits=20, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifier = GaussianNB()
        classifier.fit(x_train, y_train)

        predictions = classifier.predict(x_test)

        accuracy = metrics.accuracy_score(y_test, predictions)
        print("Accuracy : " + accuracy.__str__())

        #90%이상은 평균적으로 봤을때 일반화 되어 있다라고 한다.

        #데이터를 어떻게 잘라야할까???????
        #순서가 있가 있냐 없냐 (순서 여부)
        #shuffle을 했을시에 시간의 개념이 들어간다면 조금 어려워 진다.
