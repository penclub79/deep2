import numpy as np

xy = np.loadtxt('test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-2]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data, len(y_data))
