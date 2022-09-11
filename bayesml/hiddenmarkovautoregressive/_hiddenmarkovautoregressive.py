# Code Author
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>
# Document Author
# Koki Kazama <kokikazama@aoni.waseda.jp>
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>

import numpy as np

class GenModel:
    def __init__(
        self, 
        *, 
        K: int = None, 
        n: int = None, 
        z_n: np.ndarray = None, 
        pi: np.ndarray = None, 
        a_jk: np.ndarray = None, 
        A: np.ndarray = None, 
        d: int = None, 
        x_n_da: np.ndarray = None, 
        theta_k: np.ndarray = None, 
        theta: np.ndarray = None, 
        x_n: float = None, 
        tau_k: float = None, 
        tau: np.ndarray = None
    ):
        pass
        # TODO: get default values
