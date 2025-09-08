import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from das_dao.pareto_front import pareto_frontier



def create_folders(
        opt_round: int = 1, # optimization round
        ):

    round_name = 'Round'+str(opt_round)
    path       = os.getcwd()
    model_folder = path+"/Results"
        
    # Check if the directory exists
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    path0 = model_folder+'/Round'+str(opt_round-1)
    if not os.path.exists(path0):
        os.makedirs(path0)
    
    path1 = model_folder+'/Round'+str(opt_round)
    if not os.path.exists(path1):
        os.makedirs(path1)
    
    path2 = model_folder+'/Round'+str(opt_round+1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    return path0, path1, path2
    


def design_space(
        
        ):
    """Define the design space of all parameters"""

    param_space = {
        # Chemical compositions in unit at%, only C, Si, Mn, Cu are designable
        'C':  np.arange(2.5, 4.01, 0.01).round(2),
        'Si': np.arange(1.5, 3.01, 0.01).round(2),
        'Mn': np.arange(0.1, 1.01, 0.01).round(2),
        'Cu': np.arange(0, 1.01,   0.01).round(2),
        
        # Processing parameters: only T1,T2,t1,t2,t3 are designable
        'T1': np.arange(1420, 1501, 1).round(0), # Tap Temperature (°C)
        'T2': np.arange(1340, 1421, 1).round(0), #'Pouring start temperature (°C)
        't1': np.arange(20, 41, 1).round(0), # Nodularization time (min)
        't2': np.arange(30, 46, 1).round(0), # Inoculation time (min)
        't3': np.arange(380, 501, 10).round(0), # Pouring time (″)
        
        # Miscrostructures: only pearlite fraction is designable
        'pearlite fraction': np.round(np.arange(5, 100, 5)), # (%)
        
        # Other chemical compositions in unit at%
        'P':  0.163,
        'S':  0.0082,
        'Cr': 0.042,
        'Sn': 0.009,
        'Mg': 0.05,
        'Sb': 0.00013,
        'Ni': 0.013,
        'Mo': 0.0044,
        'Ti': 0.017,
        'Al': 0.017,
        'V':  0.003,
        'Co': 0.004,
        'Zr': 0.002,
        'B':  0.0016,
        'Ca': 0.0022,
        'As': 0.002,
        'Pb': 0.0012,
        'Bi': 0.001,
        'Zn': 0.002,
        'Ce': 0.005,
        'La': 0.004,
        'N':  0.007,

        # Other processing parameters
        'permeablity': 172,
        'compressive strength': 161, # (kPa)
        'hot wet tensile strength': 4.6, # (kPa)
        'gas evolution': 13.2,
        'clay content': 10.88, # (%)
        'AFS fineness number': 57.84,
        'loss on ignition': 4.1, 

        # # Other miscrostructures
        # 'nodularity rate':  3,
        }

    design_params = [key for key, value in param_space.items() 
                  if hasattr(value, '__len__') and len(value) > 1]
    
    return param_space, design_params

        
        
        
def load_data(
        param_space,
        file_name='data_multi-modal',
        ):
    df = pd.read_excel(f'data/{file_name}.xlsx')
    columns = [key for key, value in param_space.items()]
    df1 = df[columns]
    
    def data_ind(data):
        # return the indices of nan in the dataset
        if data.ndim == 1:
            ind = np.where(data=='/')[0]
            data[ind] = 'nan'
            data = data.astype(float)
            ind = np.argwhere(np.isnan(data))
        elif data.ndim == 2:
            ind = np.where(data=='/')
            for i in range(len(ind[0])):
                data[ind[0][i],ind[1][i]] = 'nan'
            data = data.astype(float)
            ind = np.argwhere(np.isnan(data))
        return ind

    ind = data_ind(np.array(df1))
    ind_new = np.setdiff1d(np.arange(len(np.array(df1))), ind)


    # features and labels
    input_x = np.array(df1)[ind_new].astype(float)
    data1 = np.array(df['UTS'])[ind_new].astype(float)
    data2 = np.array(df['Ef'])[ind_new].astype(float)
    
    properties01=np.concatenate((data1.reshape(-1,1),data2.reshape(-1,1)),axis=1)
    pareto_front = pareto_frontier(properties01)
    coef_fit = np.polyfit(data1[pareto_front], data2[pareto_front], 2)
    
    y_fit = np.polyval(coef_fit, data1[pareto_front])
    
    """Define StandardScaler"""
    scaler_x = StandardScaler()
    scaler_y1 = StandardScaler()
    scaler_y2 = StandardScaler()
    X_scaled = scaler_x.fit_transform(input_x.reshape(len(input_x),input_x.shape[1]))
    y1_scaled = scaler_y1.fit_transform(data1.reshape(len(data1),1))
    y2_scaled = scaler_y2.fit_transform(data2.reshape(len(data2),1))

    
    """Visulization"""
    plt.figure()
    plt.scatter(data1,data2,label='initial data')
    plt.scatter(data1[pareto_front],data2[pareto_front],label='pareto front')
    plt.plot(data1[pareto_front], y_fit, 'g')
    plt.xlabel('Ultimate tensile strength (MPa)')
    plt.ylabel('Elongation to fracture (%)')
    plt.title(f"{file_name}")
    plt.legend()
    
    print('fano factor of Tensile strength:',np.var(data1)/np.mean(data1))
    print('fano factor of Elongation:',np.var(data2)/np.mean(data2))

    return input_x, data1, data2, coef_fit, scaler_x, scaler_y1, scaler_y2
