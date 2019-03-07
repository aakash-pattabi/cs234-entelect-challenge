import numpy as np 
import os
import json 
import random
from encapsulate_state import StateEncapsulator
from scalar_to_action import ActionMapper
from basis_functions import identity_basis, interactive_basis, actions_only_basis, actions_cubic_basis, BASIS_MAP
import pickle
import argparse
import matplotlib.pyplot as plt

player_health_indices = {
	"A" : 2, 
	"B" : 10
}

player_score_indices = {
	"A" : 4, 
	"B" : 12
}

class LSPI(object):
	def __init__(self, n_games_batch, gamma, epsilon, player, player_name, 
				 delta = 0.001, basis = identity_basis, reward_for_win = True):
		self.n_games_batch = n_games_batch
		self.gamma = gamma
		self.epsilon = epsilon
		self.delta = delta

		self.player = player
		self.opponent = "A" if player == "B" else "B"
		self.reader = StateEncapsulator(player, player_name)
		self.action_mapper = ActionMapper()

		self.s = None
		self.a = None
		self.r = None
		self.sp = None

		self.b = None
		self.B = None
		self.weights = None
		self.basis = basis

		self.reward_for_win = reward_for_win

	def ingest_batch(self, batch_dir):
		self.s, self.a, self.r, self.sp = [], [], [], []

		games = os.listdir(batch_dir)
		training_batch = random.sample(games, self.n_games_batch)
		for game_dir in training_batch:
			states = self.reader.parse_states(batch_dir + "/" + game_dir)
			self.s += states
			self.a += self.reader.parse_actions(batch_dir + "/" + game_dir)

			if self.reward_for_win:
				# Rewards are defined to be 0 for all non-terminal states and 1 only in terminal states where
				# the agent wins
				rewards = [0 for i in range(len(states))]
				opponent_health = states[-1][0][0, player_health_indices[self.opponent]]
				if opponent_health == 0:
					rewards[-1] = 1
			else:
				# Rewards are defined to be the change in score from one state to the next
				scores = [state[0][0, player_score_indices[self.player]] for state in states]
				rewards = [scores[0]] + list(np.diff(scores))

			self.r += rewards
			next_states = states[1:] + [states[-1]]
			self.sp += next_states

		return self.s, self.a, self.r, self.sp

	def update_weights_from_batch(self):
		if self.s is None:
			raise ValueError("Model tried to update weights before loading data...")

		for i in range(len(self.a)):
			self.__update_weights_from_sample(self.s[i], self.a[i], self.r[i], self.sp[i])

	# Actions are stored as triples [x, y, tower] but mapped to scalars
	def __update_weights_from_sample(self, s, a, r, sp):
		a = self.action_mapper.action_to_scalar(a)
		s_a = np.array(list(s.flatten()) + [a])
		s_a = self.basis(s_a)

		if self.B is None:
			self.B = (1.0 / self.delta) * np.eye(len(s_a))
			self.b = np.zeros(len(s_a))
			self.weights = np.zeros(len(s_a))

		sp_ap = np.array(list(sp.flatten()) + [self.__get_next_action(sp)])
		sp_ap = self.basis(sp_ap)

		# Should the last "*" be a "*" or an "@"?
		numerator = (self.B @ s_a) @ np.transpose(s_a - (self.gamma * sp_ap)) * self.B
		denominator = 1 + (np.transpose(s_a - self.gamma * sp_ap) @ self.B @ s_a)
		self.B = self.B - ((1.0*numerator)/denominator)
		self.b = self.b + (r * s_a)
		self.weights = self.B @ self.b

	# Expects as input a 3D tensor representing the state, un-flattened; returns a _scalar_ action
	def __get_next_action(self, sp):
		sp = sp.flatten()
		q_values = []
		for action in range(self.action_mapper.num_actions):
			sp_ap = np.array(list(sp) + [action])
			sp_ap = self.basis(sp_ap)
			q_values.append(np.dot(sp_ap, self.weights))

		return np.argmax(q_values)

	def train(self, batch_dir, weights_filename, min_iters):
		self.ingest_batch(batch_dir)
		self.update_weights_from_batch()
		update = True
		n_iters = 1
		norms = []

		try:
			while update: 
				tmp = np.copy(self.weights)
				self.ingest_batch(batch_dir)
				self.update_weights_from_batch()
				norm = np.linalg.norm(tmp - self.weights)
				if  norm <= self.epsilon and n_iters >= min_iters:
					update = False 
				n_iters += 1
				norms.append(norm)
				print("Iteration {} | Delta in norm {}".format(n_iters, norms[-1])) 

		except KeyboardInterrupt:
			pass

		plt.plot(range(len(norms)), norms)
		plt.xlabel("Iteration")
		plt.ylabel("Delta L2 norm")
		plt.title("Convergence of weight vector over time")
		plt.savefig("convergence.png")

		with open(weights_filename, "wb") as pkl:
			pickle.dump(self.weights, pkl)
		
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_dir", type = str)
	parser.add_argument("--weights_filename", type = str)
	parser.add_argument("--n_games_batch", nargs = "?", default = 1, type = int)
	parser.add_argument("--gamma", nargs = "?", default = 0.95, type = float)
	parser.add_argument("--epsilon", nargs = "?", default = 1e-4, type = float)
	parser.add_argument("player", nargs = "?", default = "A", type = str)
	parser.add_argument("name", nargs = "?", default = "Guido", type = str)
	parser.add_argument("delta", nargs = "?", default = 0.001, type = float)
	parser.add_argument("--basis", nargs = "?", default = "linear", type = str)
	parser.add_argument("--reward_for_win", dest = "reward_for_win", action = "store_true")
	parser.add_argument("--min_iters", nargs = "?", default = 1, type = int)
	args = parser.parse_args()

	basis = BASIS_MAP[args.basis]
	agent = LSPI(args.n_games_batch, args.gamma, args.epsilon, args.player, args.name, 
				 basis = basis, reward_for_win = args.reward_for_win)
	agent.train(args.batch_dir, args.weights_filename, args.min_iters)

if __name__ == "__main__":
	main()
