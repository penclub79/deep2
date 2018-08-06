import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 데이터 로드
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#0~9까지 output을 뽑아냄
nb_classes = 10

#모델 정의
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)


cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
# Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
# Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size) #전체사이즈(10000)/100 =100번 돌면 1번

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)#100개씩 돌면서 학습
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1),
                    'cost =', '{:.9f}'.format(avg_cost))
