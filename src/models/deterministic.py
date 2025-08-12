from src.models.bellman import BellmanModel
from typing import Callable, Dict, List
import numpy as np


class DeterministicCakeEating(BellmanModel):
    def state_action_value(self, V_interp_funcs: List[Callable],
                           state_grids: Dict[str, np.ndarray],
                           control_grids: Dict[str, np.ndarray]) -> np.ndarray:
        V_interp_func = V_interp_funcs[0]
        W_grid = state_grids['W']
        c_grid = control_grids['c']
        W_mat = W_grid[np.newaxis, :, np.newaxis] # Shape: (1, n_W, 1)
        c_mat = c_grid[np.newaxis, np.newaxis, :] # Shape: (1, 1, n_c)
        is_feasible = c_mat <= W_mat
        present_value = np.where(is_feasible, self.utility_func(c_mat), -np.inf)
        W_prime = W_mat - c_mat        
        future_value = self.beta * V_interp_func(W_prime)
        return present_value + np.where(is_feasible, future_value, 0)