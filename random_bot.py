import numpy as np 
import json 
import random
from encapsulate_state import StateEncapsulator

STATE_FILENAME = "state.json"
CONFIG_FILENAME = "bot.json"

class RandomBot(object):
	def __init__(self, player, player_name):
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

	def __generate_random_action(self):
		action_candidates = [key for key, value in self.actions.items()]
		action = random.choice(action_candidates)
		if action == "NOTHING" or self.energy < self.costs[action]:
			return 

		## Map action description to code
		action = self.actions[action]

		## Choose location knowing there is enough energy to perforn random action
		location_candidates = np.argwhere(self.state[1] == -1) if action != 3 else \
			np.argwhere(self.state[1] != -1)

		## If no legal cells for action, do nothing
		location_candidates = location_candidates[location_candidates[:,1] < 8]
		if len(location_candidates) == 0: 
			return 

		location = location_candidates[np.random.choice(range(len(location_candidates)))]

		command = str(location[1]) + "," + str(location[0]) + "," + str(action)
		self.command = command

	def write_action(self):
		self.__generate_random_action()
		outfl = open("command.txt","w")
		outfl.write(self.command)
		outfl.close()

############################################################################################

if __name__ == "__main__":
	with open(CONFIG_FILENAME, "r") as f:
		data = json.load(f)
		player_name = data["nickName"]
		player = "A" if player_name == "Guido" else "B"

	bot = RandomBot(player, player_name)
	bot.write_action()
	