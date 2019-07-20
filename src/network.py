import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
EPS = 1e-4

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            net = tf.stack([split_0, split_1, split_2_flat,
                            split_3_flat, split_4_flat, split_5], axis=1)
            net = tflearn.fully_connected(
                net, FEATURE_NUM, activation='relu')
                
            mu = tflearn.fully_connected(net, self.a_dim, activation='sigmoid')
            sigma = tflearn.fully_connected(net, self.a_dim, activation='softplus')
            value = tflearn.fully_connected(net, 1, activation='linear')
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
        self.mu, self.sigma, self.val = self.CreateNetwork(inputs=self.inputs)
        self.dist = tf.distributions.Normal(self.mu, self.sigma)
        self.real_out = self.dist.sample()
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
            - tf.reduce_mean(self.log_prob * (self.outputs - tf.stop_gradient(self.val)) \
            + self.entropy_weight * tf.reduce_mean(self.entropy)
        
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
    
    def predict(self, input):
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0, 0]

    def deterministic_predict(self, input):
        action = self.sess.run(self.mu, feed_dict={
            self.inputs: input
        })
        return action[0, 0]

    def get_entropy(self, step):
        if step < 10000:
           return 0.5
        elif step < 20000:
           return 0.3
        elif step < 30000:
           return 0.1
        elif step < 35000:
           return 0.05
        else:
           return 0.03

    def train(self, s_batch, a_batch, v_batch, epoch):
        s_batch, a_batch, v_batch = tflearn.data_utils.shuffle(s_batch, a_batch, v_batch)
        self.sess.run(self.optimize, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.outputs: v_batch, 
            self.entropy_weight: self.get_entropy(epoch)
        })

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = s_batch.shape[0]
        v_batch = critic.predict(s_batch)
        R_batch = np.zeros(r_batch.shape)

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        return R_batch