
import tensorflow as tf
from collections import *
import gym
import random
import numpy as np
from ImageProcess import ImageProcess

class DQN():

    def __init__(self, env, config):

        self.image_processor = ImageProcess()

        self.epsilon = config.INITIAL_EPSILON
        self.replay_buffer = deque()
        self.recent_history_queue = deque()

        self.action_dim = env.action_space.n
        self.state_dim = config.CNN_INPUT_HEIGHT * config.CNN_INPUT_WIDTH
        self.time_step = 0

        self.session = tf.InteractiveSession()
        self.create_newtwork(config)
        self.observe_time = 0

        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.train.summary.FileWriter(config.SUMMARY_LOG, self.session.graph)

        self.session.run(tf.initialize_all_variables())

    def create_network(self, config):

        INPUT_DEPTH = config.SERIES_LENGTH

        self.input_layer = tf.placeholder(tf.float32, [None, config.cnn_input_width, config.cnn_input_height, config.cnn_input_depth], name='state input')
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        w1 = tf.get_variable(shape=[8, 8, 4, 32], initializer=tf.truncated_normal_initializer())
        b1 = tf.constant(0.01, shape=[32])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, w1, [1,4,4,1], padding='SAME',) + b1)
        conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        w2 = tf.get_variable(shape=[4,4,32,64], initializer=tf.truncated_normal_initializer())
        b2 = tf.constant(0.01, shape=[64])

        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1,2,2,1], padding='SAME') + b2)

        w3 = tf.get_variable(shape=[3,3,64,64], initializer=tf.truncated_normal_initializer())
        b3 = tf.constant(0.01, shape=[64])

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w3, strides=[1,1,1,1], padding='SAME') + b3)

        w_fc1 = tf.get_variable(shape=[1600, 512], initializer=tf.truncated_normal_initializer())
        b_fc1 = tf.constant(0.01, shape=[512])

        conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)

        w_fc2 = tf.get_variable(shape=[512, self.action_dim], initializer=tf.truncated_normal_initializer())
        b_fc2 = tf.constant(0.01, shape=[self.action_dim])

        self.Q_value = tf.matmul(h_fc1, w_fc2) + b_fc2

        Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))


        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def train_q_network(self, config):

        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, config.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={
            self.input_layer: next_state_batch
        })

        for i in range(config.BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + config.gamma * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })

    def get_greedy_action(self, state_shadow):

        rst = self.Q_value.eval(feed_dict = {
            self.input_layer: [state_shadow]
        })[0]

        print(np.max(rst))
        return np.argmax(rst)

    def percieve(self, state_shadow, action_index, reward, state_shadow_next, done, episode, config):

        action = np.zeros(self.action_dim)
        action[action_index] = 1

        self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

        self.observe_time += 1

        if self.observe_time % 1000 and self.observe_time <= config.observe_time:
            print(self.observe_time)

        if len(self.replay_buffer) > config.replay_size:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > config.batch_size and self.observe_time > config.observe_time:
            self.train_q_network()

    def get_action(self, state_shadow, config):

        if self.epsilon >= config.FINAL_EPSILON and self.observe_time > config.OBSERVE_TIME:
            self.epsilon -= (config.initial_epsilon - config.final_epsilon) / 10000

        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = self.get_greedy_action(state_shadow)

        return action_index


