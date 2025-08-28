# Student Performance Indicator

## Overview

This repository contains an end-to-end machine learning pipeline for predicting student math scores based on various features. The project demonstrates best practices in data preprocessing, model training, evaluation, and deployment using Python, scikit-learn, and AWS Elastic Beanstalk.

---

## Table of Contents

- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Data Pipeline](#data-pipeline)
- [Model Training & Evaluation](#model-training--evaluation)
- [Web Application](#web-application)
- [Deployment (AWS Elastic Beanstalk)](#deployment-aws-elastic-beanstalk)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- Data preprocessing (imputation, scaling, encoding)
- Multiple regression models (Linear, Ridge, Lasso, SVR, Random Forest, XGBoost, CatBoost, etc.)
- Model evaluation (MAE, RMSE, R2 Score)
- Hyperparameter tuning with RandomizedSearchCV
- Interactive web app for predictions (Flask)
- Automated deployment to AWS Elastic Beanstalk via CodePipeline

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ML-Project.git
cd ML-Project
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place your dataset (e.g., `StudentsPerformance.csv`) in the `data/` directory.

---

## Data Pipeline

- **Data Transformation:**  
  Uses `ColumnTransformer` and `Pipeline` to preprocess numerical and categorical features separately.
- **Feature Engineering:**  
  Handles missing values, scales numerical features, and encodes categorical variables.

---

## Model Training & Evaluation

- Trains multiple regression models and evaluates them using MAE, RMSE, and R2 Score.
- Hyperparameter tuning is performed for selected models.
- Results are visualized and compared in Jupyter notebooks.

---

## Web Application

- Built with Flask.
- Accepts user input for features and returns predicted math scores.
- HTML templates are located in the `templates/` directory.

### Running Locally

```bash
python application.py
```
Visit `http://localhost:5000` in your browser.

---

## Deployment (AWS Elastic Beanstalk)

### Prerequisites

- AWS account
- Elastic Beanstalk CLI or AWS Console access

### Steps

1. Ensure `.ebextensions/python.config` is correctly set for your app entry point.
2. Push your code to the repository connected to AWS CodePipeline.
3. CodePipeline will build and deploy your app to Elastic Beanstalk.
4. Access your app via the provided EB domain.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

This project is licensed
