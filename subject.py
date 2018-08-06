import tensorflow as tf

# indices = [[0, 2], [1, -1]]
# depth = 3
# oh = tf.one_hot(indices, depth,
#            on_value=1.0, off_value=0.0,
#            axis=1)
#
# print(oh)
indices = [0, 2, -1, 1]
depth = 3
oh = tf.one_hot(indices, depth,
           on_value=5.0, off_value=0.0,
           axis=0)

print(oh)
