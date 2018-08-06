from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#iris데이터 로드
iris = datasets.load_iris()
#X는 iris 데이터 , y는 분류
X = iris.data
y = iris.target

print(y[:5])
#70 대 30으로 X, y를 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#데이터 전처리
sc = StandardScaler() #전처리 함수
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

print(y_pred)
print(y_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))