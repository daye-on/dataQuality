from sklearn import datasets

# 데이터 세트 읽어오기
digits = datasets.load_digits()

# 무슨 데이터인지 확인하기
# import matplotlib.pyplot as plt
# plt.matshow(digits.images[0], cmap='gray')
# plt.show()

# 화상 데이터를 배열로 한 것(numpy)
x = digits.data

# 이미지 데이터에 대한 숫자(numpy)
y = digits.target

# train data 와 test data 로 나누기
x_train, y_train = x[0::2], y[0::2] # train: 짝수행
x_test, y_test = x[1::2], y[1::2]   # test: 홀수행

# SVM 알고리즘으로 학습
from sklearn import svm
clf = svm.SVC(gamma=0.001)
clf.fit(x_train, y_train)

# test data 로 정답률 반환
accuracy = clf.score(x_test, y_test)
print(f"정답률: {accuracy}")

# 학습된 모델을 사용하여 test data 를 분류한 결과 반환
predicted = clf.predict(x_test)

# 자세한 결과 반환
import sklearn.metrics as metrics
print("Classification report")
print(metrics.classification_report(y_test, predicted))