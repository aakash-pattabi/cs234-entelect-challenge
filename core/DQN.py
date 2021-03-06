import tensorflow as tf
import tensorflow.contrib.layers as layers
import os
import numpy as np

from utils.env_def import Env
from core.deep_q_learning import DQN
from configs.schedule import LinearExploration, LinearSchedule

from configs.test_config import config
from least_squares_policy_iteration import LSPI
import random
from scalar_to_action import ActionMapper
from collections import deque

class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def __init__(self, env, config):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.config = config
        self.env = env

        #just using as util for now
        #self.agent =  LSPI(self.config.batch_size, 0.99, 0.01, "A", "Guido", reward_for_win=False)
        self.agent =  LSPI(self.config.batch_size, 0.99, 0.01, "A", "Guido")
        self.action_mapper = ActionMapper()
        # build model
        self.build()

    #Overriding previous functions
    #First those defined in DQN
    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        # state /= self.config.high

        return state

    #remove dependence on replay_buffer
    def update_step(self, t, lr, batch):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        #################
        #sampling without replay_buffer
        ################# 
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = batch
        # done_mask = np.zeros(len(s_batch), dtype=bool)
        # done_mask[-1]=False # assuming batch=game for now

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            # self.avg_reward_placeholder: self.avg_reward, 
            # self.max_reward_placeholder: self.max_reward, 
            # self.std_reward_placeholder: self.std_reward, 
            # self.avg_q_placeholder: self.avg_q, 
            # self.max_q_placeholder: self.max_q, 
            # self.std_q_placeholder: self.std_q, 
            # self.eval_reward_placeholder: self.eval_reward, 
        }

        loss, grad_norm, _ = self.sess.run([self.loss, self.grad_norm, 
                                                     self.train_op], feed_dict=fd)


        # tensorboard stuff
        # self.file_writer.add_summary(summary, t)
        
        return loss, grad_norm

    def model_init(self, sess):
        self.sess = sess
    #Now in QN
    ###########################################
    #Need to update to reflect how we're representing action vector
    ###########################################    
    #For tensorboard
    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        #self.avg_reward = 0.0
        #self.max_reward = 0.0
        #self.std_reward = 0.0

        self.avg_q = 0.0
        self.max_q = 0.0
        self.std_q = 0.0
        
        self.eval_reward = 0.0

    def update_averages(self,  max_q_values, q_values):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

    def scalar_actions(self, a):
        a_sc=[]
        # print(a)
        for i in range(len(a)):
            # print(str(i)+" "+str(self.action_mapper.action_to_scalar(a[i])))
                
            a_sc.append(self.action_mapper.action_to_scalar(a[i]))
        return np.array(a_sc)

    def train(self, exp_schedule, lr_schedule, batch_dir):  
        t=0
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=10000)
        self.init_averages()

        print("Loading Data")
        s, a, r, sp, dm = map(np.array, self.agent.ingest_batch(batch_dir))
        print("Num Training Examples: " + str(len(s)))
        a_sc = self.scalar_actions(a)

        print("Starting Training")
        while t < self.config.nsteps_train:
            
            #experience replay here
            idxs = np.random.randint(0, len(s)-1, self.config.batch_size)
            batch = s[idxs], a_sc[idxs], r[idxs], sp[idxs], dm[idxs]

            for state in s[idxs]:
                best_action, q_values = self.get_best_action(state)
                action                = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

            loss, grad = self.train_step(t, lr_schedule.epsilon, batch)
            t+=1
            if t%self.config.log_freq==0:
                print("Iteration: %d" % t)
                self.update_averages(max_q_values, q_values)
                print("Max q: %f" % self.max_q)
                print("Avg q: %f" % self.avg_q)
                print("Std q: %f" % self.std_q)
        print("Training Done")

        self.save(t)

    #removed inequality on schedule
    def train_step(self, t, lr, batch):
        loss, grad = 0, 0

        # perform training step
        loss, grad = self.update_step(t, lr, batch)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            print("saving")
            self.save(t)

        return loss, grad

    #not necessary if we don't care about eval_reward

    def save(self, t):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output+"."+str(t)+"/")

    def evaluate(self, env=None, num_episodes=None):
        pass

    def run(self, exp_schedule, lr_schedule, data_dir):
        self.initialize()
        self.train(exp_schedule, lr_schedule, data_dir)

    #From assignment
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        ##############################################################
        """
        TODO: 
            Add placeholders:
            Remember that we stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32
        
        (Don't change the variable names!)
        
        HINT: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            You can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################
        s_h, s_w, nchannels = self.config.state_shape

        self.s = tf.placeholder(tf.uint8, (None, nchannels, s_w, s_h))
        self.a = tf.placeholder(tf.int32, (None))
        self.r = tf.placeholder(tf.float32, (None))
        self.sp = tf.placeholder(tf.uint8, (None, nchannels, s_w, s_h))
        self.done_mask = tf.placeholder(tf.bool, (None))
        self.lr = tf.placeholder(tf.float32)
        ##############################################################
        ######################## END YOUR CODE #######################


    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            Implement a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.

        HINT: 
            - You may find the following functions useful:
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ################## 
        
        with tf.variable_scope(scope, reuse=reuse):
            out = tf.layers.conv2d(state, filters=16, kernel_size=1, strides=1, padding="same", activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            for _ in range(self.config.num_layers):
                out = tf.layers.dense(out, self.config.hidden_size, activation=tf.nn.relu)
            out = tf.layers.dense(out, num_actions)

            #out = tf.layers.conv2d(state, filters=32, kernel_size=2, strides=1, padding="same", activation=tf.nn.relu)
            #out = tf.layers.conv2d(out, filters=64, kernel_size=3, strides=1, padding="same", activation=tf.nn.relu)
            #out = tf.layers.conv2d(out, filters=64, kernel_size=4, strides=1, padding="same", activation=tf.nn.relu)
            #out = tf.layers.flatten(out)
            #out = tf.layers.dense(out, 1024, activation=tf.nn.relu)
            #out = tf.layers.dense(out, num_actions)
        ##############################################################
        ######################## END YOUR CODE #######################

        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: 
            Add an operator self.update_target_op that for each variable in
            tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

        HINT: 
            You may find the following functions useful:
                - tf.get_collection
                - tf.assign
                - tf.group (the * operator can be used to unpack a list)

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        ops = []

        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, q_scope)
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target_q_scope)
        ops.append(q_vars)
        ops.append(t_vars)

        for q in q_vars:
            q_name = q.name[q.name.find("/")+1:]
            for t in t_vars:
                t_name = t.name[t.name.find("/")+1:]
                if q_name==t_name:
                    ops.append(tf.assign(t, q))
        # print(ops)
        self.update_target_op = tf.group(*ops)
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############
        q_samp = self.r+self.config.gamma*tf.reduce_max(target_q, axis=1)*(1-tf.cast(self.done_mask, tf.float32))

        update_locs = tf.one_hot(self.a, num_actions)
        # update_locs = tf.Print(update_locs, [tf.shape(self.a)])
        q = tf.reduce_sum(tf.multiply(update_locs, q), 1)

        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q))
        ##############################################################
        ######################## END YOUR CODE #######################


    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        ##############################################################
        """
        TODO: 
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm
             
             you can access config variables by writing self.config.variable_name
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############
        opt = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = opt.compute_gradients(self.loss, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        # grads_and_vars=tf.Print(grads_and_vars, [grads_and_vars], "grads and vars: ")
        if self.config.grad_clip:
            grads_and_vars = [ (tf.clip_by_norm(gv[0], self.config.clip_val), gv[1]) for gv in grads_and_vars]
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.grad_norm = tf.global_norm([gv[0] for gv in grads_and_vars])
        ##############################################################
        ######################## END YOUR CODE #######################
    


if __name__ == '__main__':
    # train model

    env = Env()

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    model = Linear(env, config)
    data_dir = "../2018-TowerDefence/starter-pack-ref/tower-defence-matches"
    model.run(exp_schedule, lr_schedule, data_dir)
