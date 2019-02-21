import numpy as np 
import json 
import sys
import os
import pickle
from datetime import date

class StateEncapsulator(object):
	def __init__(self, player, player_name):
		self.towers = {
			"DEFENSE" : 0,
			"ATTACK" : 1,
			"ENERGY" : 2,
			"TESLA" : 4
		}

		self.missiles = {
			"A" : 1,
			"B" : -1,
			"BOTH" : 2
		}

		self.player = player
		self.player_name = player_name
		self.state_filename = "JsonMap.json"
		self.action_filename = "PlayerCommand.txt"
		self.no_command_action = [-1, -1, -1]

	def parse_state(self, data):
		field = data["gameMap"]
		nrows = len(field)
		ncols = len(field[0])

		towers = -1 * np.ones((nrows, ncols))
		tower_healths = np.zeros((nrows, ncols))
		missiles = np.zeros((nrows, ncols))

		## Parse "gameMap" field in JSON
		for row in field:
			for cell in row:
				if cell["buildings"]:
					towers[cell["y"], cell["x"]] = self.towers[cell["buildings"][0]["buildingType"]]
					tower_healths[cell["y"], cell["x"]] = cell["buildings"][0]["health"]

				if cell["missiles"]:
					if len(cell["missiles"]) > 1:
						missiles[cell["y"], cell["x"]] = self.missiles["BOTH"]
					else:
						missiles[cell["y"], cell["x"]] = self.missiles[cell["missiles"][0]["playerType"]]

		players = data["players"]
		A = np.array([players[0]["energy"], players[0]["health"], players[0]["score"], players[0]["ironCurtainAvailable"]])
		B = np.array([players[1]["energy"], players[1]["health"], players[1]["score"], players[1]["ironCurtainAvailable"]])

		player_status = np.zeros((nrows, ncols))
		player_status[:,0:8] = np.repeat(A, 2)
		player_status[:,8:] = np.repeat(B, 2)

		state = np.zeros(tuple([4] + list(towers.shape)))
		state[0] = player_status
		state[1] = towers
		state[2] = tower_healths
		state[3] = missiles

		return state

	def parse_states(self, game_dir, write = False):				
		states = []

		rounds = os.listdir(game_dir)
		rounds.sort()

		for r in rounds:
			if "Round" in r:
				state_path = game_dir + "/" + r + "/" + self.player + " - " + self.player_name + "/" + self.state_filename

				try: 
					with open(state_path, "r") as f:
						data = json.load(f)
					state = self.parse_state(data)
					states.append(state)

				except IOError:
					print("Cannot find round or encapsulate game state:\n")
					print(state_path)
					break 
		
		if write:
			pickle_path = self.player + "_" + self.player_name + "_States_" + str(date.today()) + ".pickle"
			with open(pickle_path, "wb") as pkl:
				pickle.dump(states, pkl)

		return states

	def parse_actions(self, game_dir, write = False):
		actions = []

		rounds = os.listdir(game_dir)
		rounds.sort()

		for r in rounds:
			if "Round" in r:
				action_path = game_dir + "/" + r + "/" + self.player + " - " + self.player_name + "/" + self.action_filename

				try:
					with open(action_path, "r") as f:
						data = f.readlines()
					if len(data)>0:
						if data[0] == "No Command":
							command = self.no_command_action
						else:
							command = list(map(int, data[0].split(",")))
						actions.append(command)
				except IOError:
					print("Cannot find round or encapsulate game actions:\n")
					print(action_path)
					break 

		if write:
			pickle_path = self.player + "_" + self.player_name + "_Actions_" + str(date.today()) + ".pickle"
			
			with open(pickle_path, "wb") as pkl:
				pickle.dump(actions, pkl)
				
		return actions

############################################################################################

def main():
	reader = StateEncapsulator("A", "Guido")
	reader.parse_states("./Game 1/")	

if __name__ == "__main__":
	main()
