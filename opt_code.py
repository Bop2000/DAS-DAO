############################### Import libraries ###############################
import numpy as np
import pandas as pd
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler


from das_ai.initialize import create_folders,load_data,design_space
from das_ai.neural_surrogate import TensileSurrogateModel
from das_ai.objective_func import obj_function
from das_ai.tree_exploration import TreeExploration
from das_ai.pareto_front import pareto_frontier,pareto_evaluation,pareto_score
from das_ai.save_file import save_excel


############################### Initialization ###############################

opt_round = 1 # optimization round

# path1 is the folder of last round, path2 is this round, path3 is next round
path1, path2, path3 = create_folders( 
    opt_round = opt_round, # optimization round
    )

"set the seed for reproducibility"
random.seed(42)
np.random.seed(42)


"""Define the design space of all parameters"""
param_space, design_params = design_space()


############################### Load Data ###############################
"""
    X_input (np.ndarray): All features.
    y1 (np.ndarray): ultimate tensile strength in unit MPa.
    y2 (np.ndarray): elongation to fracture in unit %.
    coef_fit (np.ndarray): Coefficients of fitted quadratic polynomial.
    scaler_x (StandardScaler): StandardScaler for input features.
    scaler_y1 (StandardScaler): StandardScaler for label #1.
    scaler_y2 (StandardScaler): StandardScaler for label #2.
"""

"""Multi-modal dataset"""
X_input, y1, y2, coef_fit, scaler_x, scaler_y1, scaler_y2 = load_data(
    param_space,
    file_name='CPM-P_dataset',
    )


"""Composition dataset"""
columns_c = ['C','Si','Mn','Cu','P','S','Cr','Sn','Mg','Sb','Ti']
X_input_c, y1_c, y2_c, coef_fit_c, scaler_x_c, scaler_y1_c, scaler_y2_c = load_data(
    {i:param_space[i] for i in columns_c},
    file_name='C-P_dataset',
    )


######################## Surrogate model training############################
"""
Attributes for TensileSurrogateModel:
    input_dims (int): The input dimensions for the model.
    learning_rate (float): The learning rate for the optimizer.
    path (str): Where to save the trained mdoels.
    batch_size (int): The number of samples per gradient update.
    epochs (int): The number of epochs to train the model.
    patience (int): Number of epochs with no improvement after which training will be stopped.
    n_model (int): Number of models for cross-validation.
    target_R2 (float): Target R2 of model.
    try_lim (int): Max times to retrain the model to achieve target R2.
    models (dict): Load all trained models and store in the dict "models".
"""
epochs = 5000
patience = 1000
try_lim = 5


"""Surrogate model training on multi-modal dataset"""
surrogate1 = TensileSurrogateModel(
    input_dims = X_input.shape[1], 
    learning_rate = 0.001,
    path = path1,
    batch_size = 50,
    epochs = epochs,
    patience = patience,
    n_model = 5,
    target_R2 = 0.98,
    try_lim = try_lim,
    target = 'UTS',
    verbose = True,
    )
surrogate1.x_scaler = scaler_x
surrogate1.y_scaler = scaler_y1


surrogate2 = TensileSurrogateModel(
    input_dims = X_input.shape[1], 
    learning_rate = 0.001,
    path = path1,
    batch_size = 50,
    epochs = epochs,
    patience = patience,
    n_model = 5,
    target_R2 = 0.98,
    try_lim = try_lim,
    target = 'Ef',
    verbose = True
    )
surrogate2.x_scaler = scaler_x
surrogate2.y_scaler = scaler_y2


surrogate1( # train and load models, visualize and save model performance
    X_input,
    y1,
    )
surrogate2( # train and load models, visualize and save model performance
    X_input,
    y2,
    )

surrogate1.load_model() # only load models
surrogate2.load_model() # only load models



"""Surrogate model training on composition dataset"""
surrogate3 = TensileSurrogateModel(
    input_dims = X_input_c.shape[1], 
    learning_rate = 0.001,
    path = path1,
    batch_size = 50,
    epochs = epochs,
    patience = patience,
    n_model = 5,
    target_R2 = 0.98,
    try_lim = try_lim,
    target = 'UTS_c',
    verbose = True,
    )
surrogate3.x_scaler = scaler_x_c
surrogate3.y_scaler = scaler_y1_c


surrogate4 = TensileSurrogateModel(
    input_dims = X_input_c.shape[1], 
    learning_rate = 0.001,
    path = path1,
    batch_size = 50,
    epochs = epochs,
    patience = patience,
    n_model = 5,
    target_R2 = 0.98,
    try_lim = try_lim,
    target = 'Ef_c',
    verbose = True
    )
surrogate4.x_scaler = scaler_x_c
surrogate4.y_scaler = scaler_y2_c


surrogate3( # train and load models, visualize and save model performance
    X_input_c,
    y1_c,
    )
surrogate4( # train and load models, visualize and save model performance
    X_input_c,
    y2_c,
    )

surrogate3.load_model() # only load models
surrogate4.load_model() # only load models



############################### Optimization ###############################

"""
Attributes for obj_function:
    dims (int): .
    coef_fit (np.ndarray): Coefficients of Quadratic Polynomial fitted based on 
            Pareto front points of stiffness and porosity.
    
    surrogate1 (class): Ensembled surrogate models for property #1 prediction.
    surrogate2 (class): Ensembled surrogate models for property #2 prediction.
    param_space (np.ndarray): design space for all parameters.
    design_params (list): 10 designable parameters.
    
    x_scaler (StandardScaler): StandardScaler for input features.
    y_scaler1 (StandardScaler): StandardScaler for label #1.
    y_scaler2 (StandardScaler): StandardScaler for label #2.
    
"""

obj_func = obj_function(

    dims = X_input.shape[1], 
    coef_fit = coef_fit,
    
    surrogate1 = surrogate1, # ensembled surrogate models for property #1 prediction
    surrogate2 = surrogate2,
    
    param_space = param_space,
    design_params = design_params,
    
    x_scaler = surrogate1.x_scaler,
    y_scaler1 = surrogate1.y_scaler,
    y_scaler2 = surrogate2.y_scaler,
    )


"""
Attributes for TreeExploration:

    func (class): Objective function to evaluate the performance of new params.
    rollout_round (int): Rollout times, i.e. expansion times of tree search.
    ratio (float): A larger ratio can lead to more exploration less exploitation.
    num_list (list): Samples to be collected for a single tree rollout period,
        1st position number is top-scored samples, 
        2nd top-visited samples, 3rd random samples.
    n_tree (int): Num of single trees, 
        and will choose different root node for each tree.
    num_samples_per_acquisition (int): Num of final chosen samples in all trees.

"""

tree_explorer = TreeExploration(
    func = obj_func, 
    rollout_round = 200,
    ratio = 0.05,
    num_list = [5,3,1,1],
    n_tree = 10,
    num_samples_per_acquisition = 20
    )

# Perform tree exploration to find promising samples for each target relative density
input_x = np.array(X_input)
input_y = obj_func.score_eval(y1,y2)
new_xs = []
new_ys = []
for i in range(5):
    tree_explorer.func = obj_func # Set objective function for tree search
    new_x = tree_explorer.rollout(input_x, input_y)
    new_x = np.unique(new_x, axis=0)
    new_y = obj_func(new_x)
    new_xs.append(new_x)
    new_ys.append(new_y)

sample_x=np.vstack(new_xs)
sample_x=np.unique(sample_x,axis=0)
sample_pf_top=tree_explorer.pf_node(input_x,sample_x,n=100)


############### Further filter samples using composition models #################
"""Return pareto front of two scores predicted 
by multi-modal model and composition model"""

sample_pf_top_c=pareto_score(
    sample_pf_top,
    obj_func(sample_pf_top),
    param_space,
    columns_c,
    
    surrogate3,
    surrogate4,
    
    coef_fit_c,
    num=50
    )

sample_UTS, sample_Ef = obj_func(sample_pf_top_c,mode=2)

"""Visulization"""
plt.figure()
plt.scatter(y2,y1,label='initial data')
plt.scatter(sample_Ef,sample_UTS,label='AI-designs (predicted)')
plt.xlabel('Elongation to fracture (%)')
plt.ylabel('Ultimate tensile strength (MPa)')
plt.legend()

"""Save best samples"""
save_excel(sample_pf_top_c,sample_UTS,sample_Ef,param_space,path2)

 
