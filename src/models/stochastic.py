from src.models.bellman import BellmanModel
from typing import Callable, Dict, List
import numpy as np



class StochasticCakeEating(BellmanModel):
    """
    Vectorized stochastic version of the cake-eating problem.

    The state is (W, epsilon), where W is wealth and epsilon is a stochastic shock
    that affects the size of the cake in the current period.
    The Bellman equation is:
        V(W, e) = max_{c <= e*W} { u(c) + beta * E[V(W', e') | e] }
    where W' = W - c.
    """
    def state_action_value(self, V_interp_funcs: List[Callable],
                           state_grids: Dict[str, np.ndarray],
                           control_grids: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the right-hand side (RHS) of the stochastic Bellman equation.

        This method computes the value of each possible action (consumption `c`)
        from each possible state (wealth `W` and shock `epsilon`).

        Args:
            V_interp_funcs: A list of callables, where `V_interp_funcs[i]`
                            interpolates the value function for the i-th shock state.
            state_grids: A dictionary of state variable grids. Must contain 'W' and 'epsilon'.
            control_grids: A dictionary of control variable grids. Must contain 'c'.

        Returns:
            A 3D numpy array of shape (n_eps, n_W, n_c) containing the total value
            (utility + discounted expected future value) for each state-action pair.
        """
        # 1. Extract grids and parameters from the input dictionaries.
        W_grid   = state_grids['W']
        c_grid   = control_grids['c']
        eps_grid = np.array(state_grids['epsilon'])
        P = self.params['P']

        # 2. Prepare grids for vectorized operations using NumPy broadcasting.
        W_curr_broadcast = W_grid[None, :, None]      # Shape: (1, n_W, 1)
        eps_curr_broadcast = eps_grid[:, None, None]  # Shape: (n_eps, 1, 1)
        c_curr_broadcast = c_grid[None, None, :]      # Shape: (1, 1, n_c)

        # 3. Calculate current period values.
        # Effective wealth depends on the current shock. Shape: (n_eps, n_W, 1)
        available_wealth = eps_curr_broadcast * W_curr_broadcast
        is_feasible = c_curr_broadcast <= available_wealth
        present_value = np.where(is_feasible, self.utility_func(c_curr_broadcast), -np.inf)

        # 4. Calculate next period's state variable (wealth).
        W_prime = available_wealth - c_curr_broadcast

        # 5. Calculate the expected future value, E[V(W', e')].
        # For each possible future shock e', interpolate V(W', e').
        V_prime_per_future_shock = [V_interp(W_prime) for V_interp in V_interp_funcs]
        # Stack results into a 4D tensor: (n_future_eps, n_current_eps, n_W, n_c)
        V_prime_tensor = np.array(V_prime_per_future_shock) 
        # Contract the tensor with the transition matrix P to get the expectation.
        # 'im,mijk->ijk' sums over the future shock dimension 'm'.
        expected_future_value = np.einsum('im,mijk->ijk', P, V_prime_tensor)

        # 6. Combine present utility and discounted expected future value.
        return present_value + self.beta * expected_future_value
