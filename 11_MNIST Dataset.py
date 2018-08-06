import matplotlib.pyplot as plt
import numpy as np

# Training Data 로드
data_file = open('mnist_train.csv', 'r')

# Training Data 파일의 내용을 한줄씩 불러와서 문자열 리스트로 반환
training_data = data_file.readlines()

# Training Data의 두번째 데이터 확인
print(training_data[1])

#Training Data의 두번째 데이터를 ','로 분리
training_data_array = np.asfarray(training_data[11].split(","))

# 일렬로 늘어진 784 개의 픽셀 정보를 28 X 28 행렬로 변환
matrix = training_data_array[1:].reshape(28,28)

# 회색으로 Training Data의 두번째 데이터 숫자 확인
plt.imshow(matrix, cmap='Blues')
plt.show()
