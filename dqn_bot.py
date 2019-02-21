import numpy as np 
import json 
import random
from encapsulate_state import StateEncapsulator

from core.DQN import Linear
from utils.env_def import Env
from schedule import LinearExploration, LinearSchedule
from configs.configs import config

import subprocess

STATE_FILENAME = "state.json"
CONFIG_FILENAME = "bot.json"

class DQNBot(object):
	def __init__(self, player, player_name, config):
		self.reader = StateEncapsulator(player, player_name)

		with open(STATE_FILENAME, "r") as f:
			data = json.load(f)
		self.state = self.reader.parse_state(data)
		self.energy = np.max(self.state[0][:,0]) if player == "A" \
			else np.max(self.state[0][:,8])
		self.actions = {
			"DEFENSE" : 0,
			"ATTACK" : 1,
			"ENERGY" : 2,
			"TESLA" : 4,
			"DECONSTRUCT" : 3, 
			"NOTHING" : -1
		}
		self.costs = {
			"DEFENSE" : 30, c
			"ATTACK" : 30, 
			"ENERGY" : 20, 
			"TESLA" : 300,
			"DECONSTRUCT" : 0, 
			"NOTHING" : 0
		}
		self.command = ""

		#deep RL stuff
	    env = Env()

    	# exploration strategy
	    self.exp_schedule = LinearExploration(env, config.eps_begin, 
	            config.eps_end, config.eps_nsteps)

	    # learning rate schedule
	    self.lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
	            config.lr_nsteps)

		self.model = Linear(env, config)


	def action_from_vector(self, scalar):
		x = scalar / 8*6
		y = scalar % 18
		action = scalar / 18*

	def _generate_action(self):
		state = #code to get state from whatever directory
		best_action, _ = self.get_best_action(state)
		action         = self.exp_schedule.get_action(best_action)
		self.command = action_from_scalar(action)

	def write_action(self):
		self._generate_action()
		outfl = open("command.txt","w")
		outfl.write(self.command)
		outfl.close()

	def learn(self):
		num_games_to_train=100000
		for i in range(num_games_to_train):
			#run os command "make run" - need to be in right directory
			subprocess.run(["make", "run"])
			#find location of game data; some sort of [-1] index
			batch_dir = 
			#call train on that batch
			self.model.train_step(self.exp_schedule, self.lr_schedule, batch_dir)
############################################################################################

if __name__ == "__main__":
	with open(CONFIG_FILENAME, "r") as f:
		data = json.load(f)
		player_name = data["nickName"]
		player = "A" if player_name == "Guido" else "B"

	bot = DQNBot(player, player_name)
	bot.learn()
	