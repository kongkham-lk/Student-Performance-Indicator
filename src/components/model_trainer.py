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
from sklearn.model_selection import RandomizedSearchCV
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

            model_reports: dict = evaluate_model(models, X_train, y_train, X_test, y_test)
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
        