import tensorflow as tf
from encapsulate_state import StateEncapsulator
import argparse
import json
from core.DQN import Linear

from utils.env_def import Env
from core.deep_q_learning import DQN
from configs.schedule import LinearExploration, LinearSchedule

from configs.configs import config
from least_squares_policy_iteration import LSPI
import random
from scalar_to_action import ActionMapper
#from flask import Flask
#app = Flask(__name__)
#
#@app.route('/policy', methods=["POST"])
#def policy():
#    data = request.json
#    params = data['state']
#    state = np.array(data['arr'])
#
#    return 'Hello, World!'

if __name__ == "__main__":
    #load model here
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type = str)
    args = parser.parse_args()
    reader = StateEncapsulator("A", "Guido")

    with open("state.json", "r") as f:
        data = json.load(f)
    state = reader.parse_state(data) 
    with tf.Graph().as_default():
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            env = Env()
            exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end, config.eps_nsteps)
            lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end, config.lr_nsteps)
            model = Linear(env, config)
            model.model_init(sess)
            saver = tf.train.Saver()
            saver.restore(sess, args.model_dir)
            action = model.get_best_action(state)[0]
            print(action)	
        
#    app.run()
