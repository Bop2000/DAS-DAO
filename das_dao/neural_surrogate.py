"""
This module provides classes for training neural network models for various objective functions.
It includes an abstract base class and specific implementations for different objective functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    Lambda,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from typing import Any, Set, Optional, Dict, List
from collections import defaultdict, namedtuple

@dataclass
class SurrogateModel(ABC):
    """
    Abstract base class for surrogate model implementations.

    Attributes:
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
        n (int): Size of voxelized matrix (60,60,60).
    """

    input_dims: int = 40
    learning_rate: float = 0.001
    path: str = '/'
    check_point_path: Path = field(default_factory=lambda: Path("NN.keras"))
    batch_size: int = 50
    epochs: int = 5000
    patience: int = 1000
    n_model: int = 5
    target_R2: float = 0.98
    try_lim: int = 5
    models: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    target: str = 'UTS'
    verbose: bool = False
    
    x_scaler: Optional[StandardScaler] = field(default_factory=StandardScaler)
    y_scaler: Optional[StandardScaler] = field(default_factory=StandardScaler)
    

    @abstractmethod
    def create_model(self) -> keras.Model:
        """
        Create and return a Keras model.

        This method should be implemented by subclasses to define the specific
        architecture of the neural network model.

        Returns:
            keras.Model: The created Keras model.
        """
        pass

    def __call__(self, X,y):
        """
        Train the model on the given data.

        This method handles the entire training process, including data splitting,
        model creation, training, and evaluation.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            model_name (str): Name to save the model.

        """
        # X_processed, y_scaled = self.preprocess_data(X, y)
        
        # create a file to save model prediction results and model performance
        pd.DataFrame(np.empty(0)).to_csv(self.path +f'/model_performance_{self.target}.csv') 
        
        index_random = np.arange(X.shape[0])
        random.shuffle(index_random)
        X=X.reshape(-1,self.input_dims,1)
        
        for i in range(self.n_model):
            self.check_point_path = self.path + f"/{self.target}{i}_candidate.keras"
            
            # slice the data to 'n_model' parts
            ind=index_random[round(i*len(index_random)/self.n_model):round((1+i)*len(index_random)/self.n_model)]
            ind2=np.setdiff1d(index_random, ind)
            X_train,X_test,y_train,y_test = X[ind2],X[ind],y[ind2],y[ind]
          
            trytime = 0
            R2_1 = 0
            while R2_1 < self.target_R2 and trytime < self.try_lim:
                trytime += 1
                model, y_pred, R2, MAE = self.model_training(
                    X_train, 
                    X_test, 
                    y_train, 
                    y_test,
                    )
                model.summary()
                if R2 > R2_1:
                    R2_1 = R2
                    model.save(self.path + f'/{self.target}{i}.keras')
                    self.evaluate_model(
                        y_test, 
                        y_pred, 
                        filename = f'{self.target}{i}',
                        save_file=True)
        self.load_model()
    
    # def preprocess_data(self, X, y):
    #     """Standardize and reshape input/output data"""
    #     # Reshape for scaling
    #     original_shape = X.shape
    #     X_flat = X.reshape(len(y), self.input_dims)  # Flatten spatial dimensions

    #     # Standardize data
    #     X_scaled = self.x_scaler.fit_transform(X_flat)
    #     y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))

    #     # Reshape to appropriate dimensions
    #     X_processed = X_scaled.reshape(
    #         (X.shape[0], self.input_dims)
    #     )
    #     return X_processed, y_scaled.flatten()
    
    def load_model(self):
        # load all models and store in the dict "models"
        for i in range(self.n_model):
            self.models[f'model{i}']= keras.models.load_model(self.path+f'/{self.target}{i}.keras')

    # ensemble all models to predict
    def ensemble_pred(self, X):
        X_scaled=self.x_scaler.transform(X.reshape(len(X),self.input_dims))
        pred_all=0
        for i in range(self.n_model):
            temp=self.models[f'model{i}'].predict(
                X_scaled.reshape(len(X_scaled),-1,1), verbose = 0)
            temp_virgin = self.y_scaler.inverse_transform(temp.reshape(len(temp),1))
            pred_all+=temp_virgin.flatten()
        pred_all/=self.n_model
        
        return pred_all 

    def model_training(self, x_train, x_test, y_train, y_test):
        """
        Train the model on the given data.

        This method handles the entire training process, including 
        model creation, training, and evaluation.

        Args:
            x_train (np.ndarray): Input features of train-set.
            x_train (np.ndarray): Input features of test-set.
            y_train (np.ndarray): Target values of train-set.
            y_test (np.ndarray): Target values of test-set.

        Returns:
            keras.Model: The trained Keras model and its metrics.
        """
        self.model = self.create_model()
        
        X1 = self.x_scaler.transform(x_train.reshape(len(y_train),self.input_dims))
        y1 = self.y_scaler.transform(y_train.reshape(len(y_train),1))
        X2 = self.x_scaler.transform(x_test.reshape(len(y_test),self.input_dims))
        y2 = self.y_scaler.transform(y_test.reshape(len(y_test),1))
        
       
        mc = ModelCheckpoint(
            self.check_point_path,
            monitor="val_loss",
            mode="min",
            verbose=self.verbose,
            save_best_only=True,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        self.model.fit(
            X1.reshape(len(X1), self.input_dims, 1),
            y1.flatten(),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X2.reshape(len(X2),self.input_dims,1), y2.flatten()),
            callbacks=[early_stop, mc],
            verbose=self.verbose,
        )

        self.model = keras.models.load_model(self.check_point_path)
        y_pred = self.model.predict(
            X2.reshape(len(X2),self.input_dims,1), 
            verbose=self.verbose
        )
        
        y_pred2 = self.y_scaler.inverse_transform(y_pred.reshape(len(y_pred), 1))
        
        r_squared, mae = self.evaluate_model(y_test, y_pred2.flatten())

        return self.model, y_pred2, r_squared, mae


    def evaluate_model(self, y_test, y_pred, filename=None, save_file=False):
        """
        Evaluate the model's performance and plot results.

        This method calculates various performance metrics and creates a regression plot.

        Args:
            y_test (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        """
        # Calculate metrics
        metrics_dict = {
            'R': stats.pearsonr(y_pred.flatten(), y_test.flatten())[0],
            'R²': metrics.r2_score(y_test, y_pred),
            'MAE': metrics.mean_absolute_error(y_test, y_pred),
            'MSE': metrics.mean_squared_error(y_test, y_pred),
            'MAPE': metrics.mean_absolute_percentage_error(y_test, y_pred)
        }

        # Visualization
        ub = max(max(y_test.flatten()), max(y_pred.flatten()))
        lb = min(min(y_test.flatten()), min(y_pred.flatten()))
        u2l = ub - lb
        ub += 0.1 * u2l
        lb -= 0.1 * u2l
        label = {k: f"{v:.4f}" for k, v in metrics_dict.items()}
        print(f'{label}')
        
        if save_file:
        
            plt.figure(figsize=(6, 6))
            sns.regplot(x=y_pred.flatten(), y=y_test.flatten(), 
                        scatter_kws={'alpha':0.4}, line_kws={'color':'red'},
                        label = f'{label}')
            plt.title(f'{label}')
            plt.xlabel('Predicted Values')
            plt.ylabel('Actual Values')
            plt.xlim(lb,ub)
            plt.ylim(lb,ub)
            plt.grid(True)
            # plt.legend()
            plt.show()
            plt.savefig(self.path + f'/regplot_{filename}.png')
            
            perform_list = pd.read_csv(self.path + f'/model_performance_{self.target}.csv')
            y_test = pd.DataFrame(y_test)
            y_test.columns= ['ground truth']
            y_pred = pd.DataFrame(y_pred)
            y_pred.columns= ['pred']
            metric = pd.DataFrame([metrics_dict['R'], 
                                   metrics_dict['R²'], 
                                   metrics_dict['MAE'],
                                   metrics_dict['MSE'],
                                   metrics_dict['MAPE']
                                   ])
            metric.columns= ['R&R2&MAE&MSE&MAPE']
            perform_list2=pd.concat((perform_list,y_test,y_pred,metric),axis=1)
            perform_list2.drop([perform_list2.columns[0]],axis=1, inplace=True)
            perform_list2.to_csv(self.path + f'/model_performance_{self.target}.csv')

        return metrics_dict['R²'], metrics_dict['MAE']
        



class TensileSurrogateModel(SurrogateModel):
    """
    Surrogate model implementation for UTS and Ef prediction.
    """

    def create_model(self) -> keras.Model:
        model = Sequential([
            layers.Conv1D(
                filters=128,
                kernel_size=3,
                strides=1,
                padding='same', 
                activation='elu', 
                input_shape=(self.input_dims,1)),
            layers.LayerNormalization(),
            layers.Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same', 
                activation='elu'),
            layers.Dropout(0.1),
            layers.Conv1D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding='same', 
                activation='elu'),
            layers.Dropout(0.1),
            layers.Conv1D(
                filters=16,
                kernel_size=3,
                strides=1,
                padding='same', 
                activation='elu'),
            layers.Conv1D(
                filters=8,
                kernel_size=3,
                strides=1,
                padding='same', 
                activation='elu'),
            layers.Flatten(),
            Dense(64, activation='elu'),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), 
            loss='mse', metrics=["mean_squared_error"]
        )
        model.summary()
        return model


class DefaultSurrogateModel(SurrogateModel):
    """
    Default surrogate model implementation.
    """

    def create_model(self) -> keras.Model:
        model = Sequential(
            [
                Conv1D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="relu",
                    input_shape=(self.input_dims, 1),
                ),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(64, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2),
                Dropout(0.2),
                Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
                MaxPooling1D(pool_size=2, strides=1),
                Conv1D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(8, kernel_size=3, strides=1, padding="same", activation="relu"),
                Conv1D(4, kernel_size=3, strides=1, padding="same", activation="relu"),
                Flatten(),
                Dense(64, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error"
        )
        return model


class PredefinedSurrogateModel(Enum):
    TENSILE = auto()
    DEFAULT = auto()


def get_surrogate_model(
    f: PredefinedSurrogateModel = None,
) -> SurrogateModel:
    """
    Factory function to get the appropriate SurrogateModel.

    Args:
        f (str): The name of the optimization function.

    Returns:
        SurrogateModel: An instance of the appropriate SurrogateModel subclass.
    """
    model_classes = {
        PredefinedSurrogateModel.TENSILE: TensileSurrogateModel,
    }
    return model_classes.get(f, DefaultSurrogateModel)
