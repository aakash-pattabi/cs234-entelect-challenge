import numpy as np 
import os
import json 
import random
from encapsulate_state import StateEncapsulator

player_health_indices = {
	"A" : 1, 
	"B" : 9
}

class LSPI(object):
	def __init__(self, n_games_batch, gamma, epsilon, player, player_name, delta = 0.01):
		self.n_games_batch = n_games_batch
		self.gamma = gamma
		self.epsilon = epsilon
		self.delta = delta

		self.opponent = "A" if player == "B" else "B"
		self.reader = StateEncapsulator(player, player_name)

		self.s = None
		self.a = None
		self.r = None
		self.sp = None

		self.b = None
		self.B = None
		self.weights = None

	def ingest_batch(self, batch_dir):
		self.s, self.a, self.r, self.sp = [], [], [], []

		games = os.listdir(batch_dir)
		training_batch = random.sample(games, self.n_games_batch)
		for game_dir in training_batch:
			states = self.reader.parse_states(batch_dir + "/" + game_dir)
			self.s += states
			self.a += self.reader.parse_actions(batch_dir + "/" + game_dir)
			rewards = [0 for i in range(len(states))]

			## Rewards are defined to be 0 for all non-terminal states and 1 only in terminal states where
			## the agent wins. Maybe change to the delta in score from the previous timeframe? 
			opponent_health = states[-1][0][0, player_health_indices[self.opponent]]
			if opponent_health == 0:
				rewards[-1] = 1

			self.r += rewards
			next_states = states[1:] + [states[-1]]
			self.sp += next_states

		return (self.s, self.a, self.r, self.sp)

	def update_weights_from_batch(self, batch_dir):
		if self.s is None:
			raise ValueError("Model tried to update weights before loading data...")

		for i in range(len(self.s)):
			self.update_weights_from_sample(self.s[i], self.a[i], self.r[i], self.sp[i])

	def update_weights_from_sample(self, s, a, r, sp):
		s_a = np.array(list(np.flatten(s)) + a)
		sp_ap = np.array(list(np.flatten(s)) + self.get_next_action(sp))

		if self.B is None:
			self.B = (1.0/self.delta) * np.eye(len(s_a))
			self.b = np.zeros(len(s_a))

		numerator = (self.B @ s_a) @ np.transpose((s_a - self.gamma * sp_ap)) @ self.B
		denominator = 1 + (np.transpose(s_a - self.gamma * sp_ap) @ self.B @ s_a)
		self.B = self.B - ((1.0*numerator)/denominator)
		self.b = self.b + r*s_a

		self.weights = self.B @ self.b

	## Gotta be a more efficient way to do this... 
	def get_next_action(self, sp):
		pass

if __name__ == "__main__":
	agent = LSPI(1, 0.99, 0.01, "A", "Guido")
	agent.update_weights_from_batch("./Games")



