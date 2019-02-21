import numpy as np
from utils.env_def import Env

class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """
        ##############################################################
        """
        TODO: modify self.epsilon such that 
			  it is a linear interpolation from self.eps_begin to 
			  self.eps_end as t goes from 0 to self.nsteps
			  For t > self.nsteps self.epsilon remains constant
        """
        ##############################################################
        ################ YOUR CODE HERE - 3-4 lines ################## 

        frac = float(t)/self.nsteps if t<self.nsteps else 1.0
        self.epsilon = self.eps_begin-frac*(self.eps_begin-self.eps_end)

        ##############################################################
        ######################## END YOUR CODE ############## ########


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int 
                best action according some policy
        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.epsilon, return a random action
                else, return best_action

                you can access the environment via self.env

                you may use env.action_space.sample() to generate 
                a random action        
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines ##################

        return self.env.action_space.sample() if self.epsilon > np.random.uniform(0.0, 1.0) else best_action

        ##############################################################
        ######################## END YOUR CODE #######################