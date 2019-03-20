from core.DDQN import DDQN
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from utils.env_def import Env
from configs.schedule import LinearExploration, LinearSchedule

from configs.DDDQN_configs import config

class DDDQN(DDQN):

    def __init__(self, env, config):
        super().__init__(env, config)

    def get_q_values_op(self, state, scope, reuse=False):
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse=reuse):
            out = tf.layers.conv2d(state, filters=16, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            for _ in range(self.config.num_layers):
                out = tf.layers.dense(out, self.config.hidden_size, activation=tf.nn.relu)

            value = tf.layers.dense(out, self.config.hidden_size, activation=tf.nn.relu)
            value = tf.layers.dense(value, 1)

            adv = tf.layers.dense(out, self.config.hidden_size, activation=tf.nn.relu)
            adv = tf.layers.dense(adv, num_actions)

            out = value + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

            return out


if __name__ == '__main__':
    # train model

    env = Env()

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    model = DDDQN(env, config)
    data_dir = "../2018-TowerDefence/starter-pack-ref/tower-defence-matches"
    model.run(exp_schedule, lr_schedule, data_dir)
