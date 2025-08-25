import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path: str, obj: object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f'Object saved at {file_path}')
    except Exception as e:
        raise Exception(f"Error saving object: {e}")

def load_object(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise Exception(f"Error loading object: {e}")

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            name = list(models.keys())[i]
            model = models[name]
            param = params[name]
            # print("\n", "="*150, "\n", name, ":", param, "\n", "="*150, "\n")
            
            rs = RandomizedSearchCV(model, param_distributions=param, cv=5, n_jobs=-1)
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train) # Train model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate Train and Test dataset
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)

            report[name] = r2_test
            
        return report
    except Exception as e:
        raise CustomException(e, sys)