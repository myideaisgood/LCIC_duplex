import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

def model(X, layer_num, hidden_unit, scope):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        hidden = tf.layers.dense(X, hidden_unit, activation=tf.nn.relu, kernel_initializer=xavier_initializer())

        for i in range(layer_num - 1):
            hidden = tf.layers.dense(hidden, hidden_unit, activation=tf.nn.relu, kernel_initializer=xavier_initializer())

        out = tf.layers.dense(hidden, 2, kernel_initializer=xavier_initializer())

    return out, hidden