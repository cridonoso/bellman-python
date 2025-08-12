import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

class BellmanModel(ABC):
    """
    Abstract base class for defining a dynamic programming problem.
    Any specific model (e.g., Cake Eating) must inherit from this class
    and implement its methods.
    """
    def __init__(self, beta: float = 0.9,
                 utility_func: Optional[Callable] = None,
                 params: Optional[Dict] = None):
        self.beta = beta
        self.utility_func = utility_func
        self.params = params or {}

    @abstractmethod
    def state_action_value(self, 
                           V_interp: List[Callable],
                           state_grids: Dict[str, np.ndarray],
                           control_grids: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculates the right-hand side (RHS) value of the Bellman equation
        for all given states and controls.
        
        Args:
            V_interp: A function (or list of functions) that interpolates the value function V(W').
            state_grids: A dictionary representing the state grids.
            control_grids: A dictionary representing the control grids.

        Returns:
            The action value (present utility + expected future value) for all
            state-action pairs.
        """
        pass
