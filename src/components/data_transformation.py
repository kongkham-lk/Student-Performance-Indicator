import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    prepocessing_object_file_path: str = os.path.join('artifacts', 'preprocessing.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
        logging.info('Data transformation object initiated')
        try:
            logging.info(f'Created Pipelines')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('oneHotEncoder', OneHotEncoder(drop='first')),
                    ('scaler', StandardScaler(with_mean=False))  # Avoid mean centering for sparse data
                ]
            )

            logging.info(f'Created ColumnTransformer')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, num_features),
                    ('cat', cat_pipeline, cat_features)
                ],
                remainder='passthrough'
            )

            logging.info(f'Preprocessor created successfully')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info('Data transformation process started')
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test data loaded successfully')

            logging.info('Obtaining preprocessing object')
            tar_feature = 'math score'
            num_features = train_df.select_dtypes(exclude=['object']).columns
            num_features = num_features.drop(tar_feature, errors='ignore')
            cat_features = train_df.select_dtypes(include=['object']).columns
            preprocessor = self.get_data_transformer_object(num_features, cat_features)

            input_feature_train_df = train_df.drop(columns=[tar_feature], axis=1)
            target_feature_train_df = train_df[tar_feature]
            input_feature_test_df = test_df.drop(columns=[tar_feature], axis=1)
            target_feature_test_df = test_df[tar_feature]

            logging.info('Applying preprocessing object on training and testing datasets')
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info('Saving preprocessing object')

            save_object(
                file_path=self.transformation_config.prepocessing_object_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.prepocessing_object_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)