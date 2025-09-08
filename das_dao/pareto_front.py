import numpy as np
import pandas as pd

def pareto_frontier(data):
    """
    Find the Pareto frontier from a two-dimensional array.

    :param data: A two-dimensional numpy array where rows are points.
    :return: A numpy array with the points on the Pareto frontier.
    """
    # Sort data by the first dimension (x)
    indices = np.argsort(data[:, 0])
    indices = indices[::-1]
    data_sorted = data[data[:, 0].argsort()]
    data_sorted = data_sorted[::-1]
    pareto_front = [data_sorted[0]]
    pareto_indices = [indices[0]]
    for i, point in enumerate(data_sorted[1:]):
        if point[1] > pareto_front[-1][1]:  # Compare with the last point in the Pareto front
            pareto_front.append(point)
            pareto_indices.append(indices[i + 1])
    return np.array(pareto_indices)


def pareto_evaluation(X_scaled_init, X_scaled_sample,sample_score, num): 
    """pareto front of Euclidean distance + pred score"""
    sample_dist=[] # nearest neighbor distance
    for i in X_scaled_sample:
        dist_temp=1000000000000
        for n in X_scaled_init:
            dist= np.linalg.norm(i - n)
            if dist < dist_temp:
                dist_temp = round(dist,10)
        sample_dist.append(dist_temp)
    sample_dist=np.array(sample_dist)
    data=np.concatenate((sample_dist.reshape(-1,1),sample_score.reshape(-1,1)),axis=1)
    pareto_front = pareto_frontier(data)
    while len(pareto_front) < num:
        remaining_data = np.delete(data, pareto_front, axis=0)
        remaining_indices = np.delete(np.arange(data.shape[0]), pareto_front)
        pareto_front2 = pareto_frontier(remaining_data)
        pareto_front = np.concatenate((pareto_front,remaining_indices[pareto_front2]))
    ind = np.random.choice(pareto_front,num,replace=False)
    return ind



def pareto_score(
        X_sample,
        sample_score1,
        
        param_space,
        columns_c,
        
        surrogate3,
        surrogate4,
        
        coef_fit_c,
        
        num,
        ): 
    """pareto front of two predicted scores 
    by multi-modal model and composition model"""
    
    sample_df=pd.DataFrame(X_sample)
    sample_df.columns=[key for key, value in param_space.items()]
    sample_df_c=sample_df[columns_c]
    UTS = surrogate3.ensemble_pred(np.array(sample_df_c)).flatten()
    Ef = surrogate4.ensemble_pred(np.array(sample_df_c)).flatten()
    y_fit = np.polyval(coef_fit_c, UTS)
    sample_score2 = Ef/y_fit
    
    data=np.concatenate((sample_score2.reshape(-1,1),sample_score1.reshape(-1,1)),axis=1)
    pareto_front = pareto_frontier(data)
    while len(pareto_front) < num:
        remaining_data = np.delete(data, pareto_front, axis=0)
        remaining_indices = np.delete(np.arange(data.shape[0]), pareto_front)
        pareto_front2 = pareto_frontier(remaining_data)
        pareto_front = np.concatenate((pareto_front,remaining_indices[pareto_front2]))
    ind = np.random.choice(pareto_front,num,replace=False)
    return X_sample[ind]