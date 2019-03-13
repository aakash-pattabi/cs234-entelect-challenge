import numpy as np 

class ActionMapper(object):
	def __init__(self):
		self.scalar_to_triple = None
		self.triple_to_scalar = None
		self.num_actions = None
		self.no_command_action = [-1, -1, -1]
		self.triples = None
		self.init_action_maps()

	def init_action_maps(self):
		action_keys = {
			"DEFENSE" : 0,
			"ATTACK" : 1,
			"ENERGY" : 2,
			"TESLA" : 4,
			"DECONSTRUCT" : 3
		}

		actions = np.array([action_keys[key] for key in action_keys])
		grid_rows = np.array(range(8))
		grid_cols = np.array(range(8))
		self.num_actions = len(actions)*len(grid_rows)*len(grid_cols)

		doubles = np.transpose([np.tile(grid_rows, len(grid_cols)), np.repeat(grid_cols, len(grid_rows))])
		triples = np.reshape(np.tile(doubles, len(actions)), (self.num_actions, 2))
		triple_actions = np.expand_dims(np.tile(actions, len(grid_rows)*len(grid_cols)), 1)
		triples = np.hstack((triples, triple_actions))
		self.triples = [tuple(triples[i]) for i in range(triples.shape[0])]
		self.scalar_to_triple = {i : triples[i] for i in range(triples.shape[0])}
		self.scalar_to_triple[-1] = self.no_command_action
		self.triple_to_scalar = {tuple(value) : key for key, value in self.scalar_to_triple.items()}

	def scalar_to_action(self, action_scalar):
		return self.scalar_to_triple[action_scalar]

	def action_to_scalar(self, action_list):
		return self.triple_to_scalar[tuple(action_list)]