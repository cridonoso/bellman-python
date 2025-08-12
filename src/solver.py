import numpy as np

from scipy.interpolate import interp1d
from src.models import BellmanModel
from tqdm import tqdm


class ValueFunctionIterator:
    """
    Generic solver that solves any `BellmanModel` using value function iteration.
    """
    def __init__(self, model: BellmanModel):
        if not isinstance(model, BellmanModel):
            raise TypeError("The provided model must be an instance of BellmanModel.")
        self.model = model

    def solve(self, state_grids, control_grids, tolerance=1e-6, max_iter=1000):
        """
        Solves the dynamic programming model using value function iteration.

        Args:
            state_grids (dict): A dictionary of state variable grids. Must contain 'W'.
            control_grids (dict): A dictionary of control variable grids. Must contain 'c'.
            tolerance (float, optional): The convergence tolerance for the value function. 
                                         Defaults to 1e-6.
            max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

        Returns:
            dict: A dictionary containing the solution, including the value function ('V'),
                  the consumption policy ('cpol'), the resulting wealth policy ('wpol'),
                  and the iteration history ('history').
        """
        other_states   = {k: v for k, v in state_grids.items() if k != 'W'}
        n_other_states = len(list(other_states.values())[0]) if other_states else 1
        
        main_state_grid = state_grids['W']
        control_grid    = control_grids['c']
        history = {'V': [], 'distance': []} 

        V = np.zeros((n_other_states, len(main_state_grid)))
        pbar = tqdm(range(max_iter), desc='Iterating on V')
        for i in pbar:
            V_old = V.copy()

            V_interp_funcs = []
            for s in range(n_other_states):
                curr_fn = interp1d(main_state_grid, V[s, :],
                                   kind='linear', 
                                   fill_value=0., 
                                   bounds_error=False) 
                V_interp_funcs.append(curr_fn)
            
            rhs_tensor = self.model.state_action_value(
                V_interp_funcs, state_grids, control_grids
            )
   
            V = np.max(rhs_tensor, axis=-1)
            V = np.nan_to_num(V, neginf=0.0)
            distance = np.max(np.abs(V - V_old))
            
            pbar.set_postfix(dist=f'{distance:.2e}')
            
            history['V'].append(V)
            history['distance'].append(distance)

            if distance < tolerance:
                pbar.set_description(f"Converged in {i+1} iterations")
                break
        else:
            pbar.set_description(f"Max iter ({max_iter}) reached")
        
        policy_indices = np.argmax(rhs_tensor, axis=-1)
        policy_c = control_grid[policy_indices]

        if other_states:
            available_wealth = state_grids['epsilon'][:, None] * main_state_grid
            policy_W = available_wealth - policy_c
        else:
            policy_W = main_state_grid - policy_c

        return {'states':state_grids, 
                'controls':control_grids, 
                'V': np.squeeze(V), 
                'cpol': np.squeeze(policy_c),
                'wpol': np.squeeze(policy_W),
                'history':history}