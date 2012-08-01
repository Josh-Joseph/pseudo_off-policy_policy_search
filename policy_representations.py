import numpy as np
import scipy.spatial
import rl_tools



class Discrete_policy():
    def __init__(self, state_centers, action_centers, states_to_actions):
        self.state_centers = state_centers
        self.action_centers = action_centers
        self.states_to_actions = states_to_actions
        self.dim_centers = rl_tools.split_states_on_dim(state_centers)

    def get_action(self, s):
        return self.action_centers[self.states_to_actions[rl_tools.find_nearest_index_fast(self.dim_centers, s)]]