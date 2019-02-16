
import tensorflow as tf
from collections import *
import gym
import random
import numpy as np
from ImageProcess import ImageProcess

class DQN():

    def __init__(self, env, config):

        self.image_processor = ImageProcess()

        self.epsilon = config.getfloat('agent','initial_epsilon')
        self.replay_buffer = deque()
        self.recent_history_queue = deque()

        self.action_dim = env.action_space.n
        self.state_dim = config.getint('agent', 'cnn_input_height') * config.getint('agent', 'cnn_input_width')
        self.time_step = 0

        self.session = tf.Session()
        self.create_network(config)
        self.observe_time = 0

        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(config.get('agent', 'summary_log'),
                                                          self.session.graph)

        self.session.run(tf.initialize_all_variables())

    def create_network(self, config):


        self.input_layer = tf.placeholder(tf.float32, [None, config.get('agent','cnn_input_width'),
                                                       config.get('agent', 'cnn_input_height'),
                                                       config.get('agent', 'serie_length')],
                                          name='state_input')

        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        w1 = tf.get_variable(shape=[8, 8, 4, 32], initializer=tf.truncated_normal_initializer(),
                             name='w1')
        b1 = tf.constant(0.0, shape=[32], name='b1')

        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, w1, [1,4,4,1], padding='SAME') + b1)
        self.tmp4 = h_conv1
        self.tmp5 = w1
        conv1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        self.tmp3 = conv1
        w2 = tf.get_variable(shape=[4,4,32,64], initializer=tf.truncated_normal_initializer(),
                             name='w2')
        b2 = tf.constant(0.0, shape=[64], name='b2')

        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w2, strides=[1,2,2,1], padding='SAME') + b2)

        w3 = tf.get_variable(shape=[3,3,64,64], initializer=tf.truncated_normal_initializer(),
                             name='w3')
        b3 = tf.constant(0.0, shape=[64], name='b3')

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w3, strides=[1,1,1,1], padding='SAME') + b3)

        self.tmp = h_conv3
        w_fc1 = tf.get_variable(shape=[8960, 512], initializer=tf.truncated_normal_initializer(),
                                name='wfc1')
        b_fc1 = tf.constant(0., shape=[512], name='bfc1')


        conv3_flat = tf.reshape(h_conv3, [-1, 8960])

        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, w_fc1) + b_fc1)
        self.tmp1 = h_fc1
        w_fc2 = tf.get_variable(shape=[512, self.action_dim], initializer=tf.truncated_normal_initializer(),
                                name='wfc2')
        b_fc2 = tf.constant(0.0, shape=[self.action_dim], name='bfc2')
        self.tmp2 = w_fc2
        self.Q_value = tf.matmul(h_fc1, w_fc2) + b_fc2

        self.Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.Q_action))


        self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

        tf.summary.scalar('loss', self.cost)
        tf.summary.histogram('Q_value', self.Q_value)

        # self.episode_reward = tf.placeholder(dtype=tf.float32, shape=[1], name="reward")
        # self.reward_summary = tf.summary.scalar(tensor=self.episode_reward, name="episode_rewaord")

        self.sum_ops = tf.summary.merge_all()


    def train_q_network(self, config):

        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, config.getint('agent','batch_size'))
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        y_batch = []
        Q_value_batch = self.session.run(self.Q_value, feed_dict={
            self.input_layer: next_state_batch
        })
        for i in range(config.getint('agent','batch_size')):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + config.getfloat('agent', 'gamma') * np.max(Q_value_batch[i]))


        # print("TD Target:")
        # print(y_batch)
        # print("Q: ")
        q_value = self.session.run(self.Q_value, feed_dict={
            self.input_layer: state_batch,
        })
        print(q_value)
        print("="*100)
        a, b, c, d = self.session.run([self.tmp4, self.tmp5, self.tmp2, self.tmp3], feed_dict={
            self.input_layer: state_batch
        })
        print(a)
        print("#" * 100)
        print(b)
        print("#" * 100)
        print(state_batch[0].shape)
        print(state_batch[0][100:120, 30:40, 1])
        print("*"*100)

        self.session.run(self.optimizer, feed_dict={
            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })


        log = self.session.run(self.sum_ops, feed_dict={
            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })
        self.summary_writer.add_summary(log, global_step=self.time_step)


    def get_greedy_action(self, state_shadow):

        rst = self.session.run(self.Q_value, feed_dict = {
            self.input_layer: [state_shadow]
        })[0]

        return np.argmax(rst)

    def percieve(self, state_shadow, action_index, reward, state_shadow_next, done, config):

        action = np.zeros(self.action_dim)
        action[action_index] = 1

        self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

        self.observe_time += 1

        if self.observe_time % 1000 and self.observe_time <= config.getint('agent', 'observe_time'):
            # print(self.observe_time)
            pass

        if len(self.replay_buffer) > config.getint('agent', 'replay_size'):
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > config.getint('agent', 'batch_size') \
                and self.observe_time > config.getint('agent', 'observe_time'):
            self.train_q_network(config)

    def get_action(self, state_shadow, config):

        if self.epsilon >= config.getfloat('agent', 'final_epsilon') \
                and self.observe_time > config.getfloat('agent', 'observe_time'):
            self.epsilon -= (config.getfloat('agent', 'initial_epsilon')
                             - config.getfloat('agent', 'final_epsilon')) / 10000

        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = self.get_greedy_action(state_shadow)

        return action_index


