import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 64
EPS = 1e-4
GAMMA = 0.99

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            w_init = tf.random_normal_initializer(0., .1)
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, :], FEATURE_NUM, weights_init = w_init, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, :], FEATURE_NUM, weights_init = w_init, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, weights_init = w_init, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, weights_init = w_init, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, weights_init = w_init, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, :], FEATURE_NUM, weights_init = w_init, activation='relu')
            split_6 = tflearn.fully_connected(
                inputs[:, 6:7, :2], FEATURE_NUM, weights_init = w_init, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            net = tf.stack([split_0, split_1, split_2_flat,
                            split_3_flat, split_4_flat, split_5, split_6], axis=1)
            net = tflearn.fully_connected(
                net, FEATURE_NUM, weights_init = w_init, activation='relu')
                
            mu = tflearn.fully_connected(net, 1, weights_init = w_init, activation='sigmoid')
            sigma = tflearn.fully_connected(net, 1, weights_init = w_init, activation='sigmoid')
            
            value = tflearn.fully_connected(net, 1, weights_init = w_init, activation='linear')
            return mu, sigma, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
        
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.outputs = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.acts = tf.placeholder(tf.float32, [None, 1])
        self.entropy_weight = tf.placeholder(tf.float32)
        self.mu_, self.sigma_, self.val = self.CreateNetwork(inputs=self.inputs)
        self.mu = tf.multiply(self.mu_, 60.)
        self.sigma = tf.multiply(self.sigma_, 60.)
        self.dist = tf.distributions.Normal(self.mu, self.sigma + 1e-2)
        self.real_out = tf.clip_by_value(tf.squeeze(self.dist.sample(1), axis=0), 0., 60.)
        self.log_prob = self.dist.log_prob(self.acts)
        self.entropy = self.dist.entropy()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = 0.5 * tflearn.mean_square(self.val, self.outputs) \
            - tf.reduce_mean(self.log_prob * (self.outputs - tf.stop_gradient(self.val))) \
            + self.entropy_weight * tf.reduce_mean(self.entropy)
        
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
    
    def predict(self, input):
        action = self.sess.run([self.real_out, self.sigma], feed_dict={
            self.inputs: input
        })
        return action[0, 0]

    def deterministic_predict(self, input):
        action = self.sess.run(self.mu, feed_dict={
            self.inputs: input
        })
        return action[0, 0]

    def get_entropy(self, step):
        if step < 20000:
            return 5.
        elif step < 50000:
            return 3.
        elif step < 70000:
            return 1.
        elif step < 90000:
            return 0.5
        elif step < 120000:
            return 0.3
        else:
            return 0.1

    def train(self, s_batch, a_batch, v_batch, epoch):
        # print s_batch.shape, a_batch.shape, v_batch.shape
        s_batch, a_batch, v_batch = tflearn.data_utils.shuffle(s_batch, a_batch, v_batch)
        self.sess.run(self.optimize, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.outputs: v_batch, 
            self.entropy_weight: self.get_entropy(epoch)
        })

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        v_batch = self.sess.run(self.val, feed_dict={
            self.inputs: s_batch
        })
        R_batch = np.zeros([len(r_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)
