from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 데이터 로드
mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# 모델 정의
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b) #bias 가중치

# 크로스 엔트로피와 옵티마이저 정의
learnin_rate = 0.5
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=learnin_rate).minimize(cross_entropy)