#######################################################################
#																	  #
# Basis functions for an RL agent trained through LSPI -- meant to    #
# break linearity in the Q-function given an (s, a) pair as an input  #
# 																	  #
# Each function takes as an input a state-action vector of size 	  #
# state_dim + 1 and outputs a transformation of that vector with the  #
# appropriate basis function applied. 								  #
#																	  #
# It is assumed that the action in the state-action vector is the 	  #
# _last_ entry. 													  #
#																	  #
#######################################################################

import numpy as np

def identity_basis(s_a):
	return s_a

def interactive_basis(s_a):
	action = s_a[-1]
	state = s_a[:-1]
	s_x_a = action * state
	s_a = list(s_a) + list(s_x_a)
	return np.array(s_a)

def actions_only_basis(s_a):
	action = s_a[-1]
	state = s_a[:-1]
	s_x_a = action * state
	return np.array(s_x_a)

def actions_cubic_basis(s_a):
	action = s_a[-1]
	state = s_a[:-1]
	s_x_a = action * state
	s_x_a2 = (action * action) * state
	s_x_a3 = (action * action * action) * state
	s_a = list(s_x_a) + list(s_x_a2) + list(s_x_a3)
	return np.array(s_a)

BASIS_MAP = {
	"identity" : identity_basis,
	"interactive" : interactive_basis,
	"actions_only" : actions_only_basis, 
	"actions_cubic" : actions_cubic_basis
}