import numpy as np
import os
import pandas as pd

def save_excel(samples,sample_UTS,sample_Ef,param_space,path):
    df_pred = pd.DataFrame(np.concatenate((samples.round(5),
                                            sample_UTS.reshape(-1,1),
                                            sample_Ef.reshape(-1,1),
                                            ),axis=1))

    df_pred.columns= [key for key, value in param_space.items()] + ['pred TS','pred EL']
    df_pred.to_excel(path+f'/top_samples.xlsx')

