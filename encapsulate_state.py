import numpy as np 
import json 
import sys

class StateEncapsulator(object):
	def __init__(self):
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

############################################################################################

def main(filename):
	reader = StateEncapsulator()

	try: 
		with open(filename, "r") as f:
			data = json.load(f)

		state = reader.parse_state(data)

	except IOError:
		print("Cannot encapsulate game state")

if __name__ == "__main__":
	main(sys.argv[1])