import sys
import os
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_train_model(self, train_arr, test_arr):
        try:
            logging.info('Model training started')
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])

            models = {
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'SVR': SVR(),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False),
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }

            params = {
                'KNeighborsRegressor': {
                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'n_neighbors': [3,5,7,9]
                },
                'DecisionTreeRegressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                'AdaBoostRegressor': {
                    'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8,16,32,64,128,256],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5]
                },
                'GradientBoostingRegressor': {
                    'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
                    'n_estimators': [8,16,32,64,128,256],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion': ['friedman_mse', 'squared_error']
                },
                'RandomForestRegressor': {
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
                },
                'SVR': {},
                'LinearRegression': {},
                'Ridge': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                    'max_iter': [1000, 5000, 10000],
                    'tol': [1e-3, 1e-4, 1e-5]
                },
                'Lasso': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 5000, 10000],
                    'tol': [1e-3, 1e-4, 1e-5],
                    'selection': ['cyclic', 'random'] 
                },
                'CatBoostRegressor': {
                    'iterations': [30,50,100],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [6,8,10],
                },
                'XGBRegressor': {
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
                    'n_estimators': [8,16,32,64,128,256],
                }
            }

            model_reports: dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            best_model_score = max(model_reports.values())
            best_model_name = list(model_reports.keys())[list(model_reports.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            logging.info(f'Best model found: {best_model_name}, with score: {best_model_score:.4f}')

            best_model = models[best_model_name]
            save_object(self.trainer_config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            return r2_score(y_test, y_pred)

        except Exception as e:
            raise CustomException(e, sys)
        