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

# 트레이닝
with tf.Session() as sess:
# Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    print("Training.......")
    for i in range(1000):
        #1000 번씩, 전체 데이터에서 100개씩 뽑아서 트레이닝을 함.
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, y_:batch_y})

    print("Testing Model")
    correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy", sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
    print('Finish')

