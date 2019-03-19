from core.DQN import Linear
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from utils.env_def import Env
from configs.schedule import LinearExploration, LinearSchedule

from configs.DDQN_configs import config

class DDQN(Linear):

    def __init__(self, env, config):
        super().__init__(env, config)

    def add_loss_op(self, q, target_q):
        num_actions = self.env.action_space.n
        max_inds = tf.argmax(q, axis=1)
        num_examples = tf.cast(tf.shape(q)[0], dtype=max_inds.dtype)
        max_inds = tf.stack([tf.range(num_examples), max_inds], axis=1)
        
        q_samp = self.r+self.config.gamma*(tf.gather_nd(target_q, max_inds))*(1-tf.cast(self.done_mask, tf.float32))

        update_locs = tf.one_hot(self.a, num_actions)
        # update_locs = tf.Print(update_locs, [tf.shape(self.a)])
        q = tf.reduce_sum(tf.multiply(update_locs, q), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q))

    #may need to modify get_best_action_op
    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        q, target_q = self.sess.run([self.q, self.target_q], feed_dict={self.s: [state], self.sp: [state]})
        action_values = q[0]+target_q[0]
        return np.argmax(action_values), action_values



if __name__ == '__main__':
    # train model

    env = Env()

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    model = DDQN(env, config)
    data_dir = "../2018-TowerDefence/starter-pack-ref/tower-defence-matches"
    model.run(exp_schedule, lr_schedule, data_dir)
