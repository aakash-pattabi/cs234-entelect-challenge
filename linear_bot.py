import numpy as np 
import json 
import random
from encapsulate_state import StateEncapsulator
from scalar_to_action import ActionMapper
import pickle
from basis_functions import identity_basis, interactive_basis, actions_only_basis, actions_cubic_basis, BASIS_MAP
import matplotlib.pyplot as plt

STATE_FILENAME = "state3.json"
CONFIG_FILENAME = "bot.json"
WEIGHTS_FILENAME = "weights.pkl"
DO_NOTHING_ACTION = [-1, -1, -1]

class LinearBot(object):
		def __init__(self, player, player_name, weights_file, basis):
			self.reader = StateEncapsulator(player, player_name)

			with open(STATE_FILENAME, "r") as f:
				data = json.load(f)
			self.state = self.reader.parse_state(data)

			with open(weights_file, "rb") as pkl:
				self.weights = pickle.load(pkl)

			self.action_mapper = ActionMapper()
			self.basis = basis
			self.command = ""

		# Expects as input a 3D tensor representing the state, un-flattened; returns a list action
		def __get_next_action(self, sp):
			sp = sp.flatten()
			q_values = []
			for action in self.action_mapper.triples:
				sp_ap = np.array(list(sp) + list(action))
				sp_ap = self.basis(sp_ap)
				q_values.append(np.dot(sp_ap, self.weights))

			return list(self.action_mapper.triples[np.argmax(q_values)])

		def write_action(self):
			action_list = self.__get_next_action(self.state)
			if (not np.all(action_list == DO_NOTHING_ACTION)) and action_list[2] != -1:
				self.command = str(action_list[0]) + "," + str(action_list[1]) + "," + str(action_list[2])
			with open("command.txt", "w") as outfl:
				outfl.write(self.command)

############################################################################################

if __name__ == "__main__":
	with open(CONFIG_FILENAME, "r") as f:
		data = json.load(f)
		player_name = data["nickName"]
		player = "A" if player_name == "Guido" else "B"

	bot = LinearBot(player, player_name, WEIGHTS_FILENAME, actions_cubic_basis)
	bot.write_action()
	