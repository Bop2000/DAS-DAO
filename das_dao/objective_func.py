import numpy as np
from tensorflow import keras

from dataclasses import dataclass, field
from typing import Any, Set, Optional, Dict, List
from sklearn.preprocessing import StandardScaler

from das_dao.neural_surrogate import TensileSurrogateModel

@dataclass
class obj_function:
    """
    Attributes:
        dims (int): Feature dimension.
        coef_fit (np.ndarray): Coefficients of Quadratic Polynomial fitted based on 
            Pareto front points of stiffness and porosity.
        
        surrogate1 (class): Ensembled surrogate models for property #1 prediction.
        surrogate2 (class): Ensembled surrogate models for property #2 prediction.
        
        input_shape (int): Shape of input parameter matrix.
        param_space (np.ndarray): design space for all parameters.
        
        x_scaler (StandardScaler): StandardScaler for input features.
        y_scaler1 (StandardScaler): StandardScaler for label #1.
        y_scaler2 (StandardScaler): StandardScaler for label #2.

    """
    
    dims: int = 40
    coef_fit: np.ndarray = field(default_factory=lambda: np.zeros(3))

    surrogate1: Optional[TensileSurrogateModel] = None
    surrogate2: Optional[TensileSurrogateModel] = None

    input_shape: int = 40
    param_space: Dict[Any, Set] = field(default_factory=dict)
    design_params: List[str] = field(default_factory=lambda: [])
        
    x_scaler: Optional[StandardScaler] = StandardScaler()
    y_scaler1: Optional[StandardScaler] = StandardScaler()
    y_scaler2: Optional[StandardScaler] = StandardScaler()

    def __call__(self, x: np.ndarray, # parameters
                 mode: int = 1, # return score (1) or property (!=1)
                 ):
        x=x.reshape(len(x),self.dims,1)
        UTS = self.surrogate1.ensemble_pred(x).flatten()
        Ef = self.surrogate2.ensemble_pred(x).flatten()
        
        self.UTS = UTS
        self.Ef = Ef
        score = self.score_eval(UTS,Ef)
        
        if mode == 1:
            return score.flatten()
        else:
            return UTS.flatten(),Ef.flatten()
        
    
    # define the standard of score
    def score_eval(self,UTS,Ef):
        y_fit = np.polyval(self.coef_fit, UTS)
        return Ef/y_fit




