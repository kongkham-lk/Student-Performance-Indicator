import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill

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
        return dill.load(file_path)
    except Exception as e:
        raise Exception(f"Error loading object: {e}")