# Stroke Prediction

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#Objectives)
3. [Conclusions](#Conclusions)
4. [Project Structure](#project-structure)
5. [Technologies Used](#technologies-used)
6. [How to Use](#how-to-use)

## Overview

This project focuses on analyzing a dataset to predict the likelihood of a patient experiencing a stroke based on various demographic, medical, and lifestyle factors. The primary objective is to identify significant risk factors contributing to stroke and build robust machine learning models that can effectively predict stroke occurrence.

The dataset includes features such as age, gender, medical history (e.g., hypertension, heart disease), lifestyle attributes (e.g., smoking status, marital status), and biometric data (e.g., average glucose levels, BMI). By leveraging statistical analysis, machine learning, and advanced evaluation techniques, the project aims to uncover critical insights and create predictive tools that can assist in stroke prevention and healthcare decision-making.

## Objectives

The main objectives of this project are as follows:

1. Identify Key Factors Contributing to Stroke Risk:

   - Analyze features such as age, gender, medical history (e.g., hypertension, heart disease), lifestyle factors (e.g., smoking status, marital status), and biometric data (e.g., BMI, average glucose levels) to identify the strongest predictors of stroke occurrence.

2. Conduct Exploratory Data Analysis (EDA):

   - Visualize the relationships between various features and the target variable (stroke occurrence) using plots such as histograms, scatter plots, box plots, and correlation matrices.
   - Investigate trends, patterns, and potential anomalies in the data.

3. Formulate and Test Hypotheses:

   - Develop hypotheses about the influence of factors like hypertension, age, and smoking on stroke risk and test them using statistical techniques such as t-tests, chi-squared tests, or ANOVA.

4. Data Cleaning and Preprocessing:

   - Handle missing values, outliers, and inconsistencies in the dataset.
   - Encode categorical variables, scale numerical features, and address potential data imbalances using techniques like SMOTE.

5. Build Predictive Models:

   - Develop machine learning models such as Logistic Regression, Random Forest, Support Vector Machines (SVM), and Gradient Boosting to predict the likelihood of a stroke.
   - Use ensembling techniques to combine the strengths of multiple models and improve prediction accuracy.

6. Hyperparameter Tuning and Optimization:

   - Optimize models using hyperparameter tuning techniques such as GridSearchCV or RandomizedSearchCV to achieve the best possible performance.

7. Evaluate Model Performance:

   - Assess models using metrics such as accuracy, precision, recall, F1-score, ROC AUC, and confusion matrices.
   - Perform evaluation on both training and hold-out test datasets to ensure models generalize well to unseen data.

8. Provide Actionable Insights:

   - Summarize findings to inform healthcare strategies and improve stroke prevention efforts.
   - Highlight the most critical risk factors and their implications for patient care and public health initiatives.

## Conclusions

Based on the analysis and modeling performed in this project, the following conclusions were drawn:

1. Key Factors Contributing to Stroke Risk:

   - Features such as age, hypertension, heart disease, and average glucose levels were identified as the most significant predictors of stroke occurrence.
   - Lifestyle factors such as smoking status and marital status also influenced stroke likelihood, with certain groups (e.g., smokers, married individuals) showing distinct risk patterns.

2. Impact of Age and Medical History:

   - Age emerged as the strongest predictor of stroke, with older individuals being at a significantly higher risk.
   - The presence of hypertension and heart disease had a substantial impact, highlighting the importance of managing these conditions to reduce stroke risk.

3. Modeling and Prediction:

   - The Random Forest and Gradient Boosting models demonstrated the best performance in predicting stroke, effectively capturing both linear and non-linear relationships within the data.
   - Hyperparameter tuning significantly enhanced model accuracy, precision, and recall, making these models reliable tools for stroke prediction.

4. Data Insights and Visualization:

   - Exploratory Data Analysis revealed clear trends in the data, such as higher stroke rates in older individuals and those with elevated glucose levels or BMI.
   - Visualizations such as box plots and scatter plots provided actionable insights into the relationships between key features and stroke occurrence.

5. Next Steps and Recommendations:
   - Public health initiatives should prioritize early detection and management of hypertension and heart disease as a means of stroke prevention.
   - Future work could incorporate additional data points, such as genetic predisposition or detailed dietary habits, to further improve predictive accuracy.
   - Advanced modeling techniques, such as deep learning or ensemble stacking, could be explored for even more robust predictions.
   - Targeted awareness campaigns and personalized healthcare interventions could be developed based on the insights derived from this analysis.

## Project Structure

The project directory contains the following files:

```
.
├── data/
│ └── healthcare-dataset-stroke-data.csv
├── models/
│   └── LogisticRegression_best_model.pkl
├── web_api/
│   ├── flask_app/
│   │   ├── templates/
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── fastapi_api/
│   │   ├── Dockerfile
│   │   ├── LogisticRegression_best_model.pkl
│   │   ├── main.py
│   │   └── requirements.txt
│   └── docker-compose.yml
├── src/
│ └── engineering_utils.py
│ └── stats_utils.py
│ └── utils.py
│ └── init.py
├── stroke_prediction.ipynb
├── .gitignore
├── README.md
└── requirements.txt

```

- data: Contains all the data files used in the project.

  - `healthcare-dataset-stroke-data.csv`: Contains the dataset used to predict the likelihood of stroke based on various patient features.

- models: Contains the saved machine learning models.

  - `LogisticRegression_best_model.pkl`: The best-tuned logistic regression model saved as a .pkl

- web_api: Contains the web API implementations using Flask and FastAPI for serving the trained model.

  web_api/flask_app/

  - `templates/`: Contains HTML templates for Flask views.
  - `app.py`: Main Flask application script for serving the model.
  - `Dockerfile`: Docker configuration for deploying the Flask application.
  - `requirements.txt`: Lists Python dependencies for the Flask application.

  web_api/fastapi_api/

  - `Dockerfile`: Docker configuration for deploying the FastAPI application.
  - `LogisticRegression_best_model.pkl`: The best-tuned logistic regression model used by the FastAPI application.
  - `main.py`: Main FastAPI application script for serving the model.
  - `requirements.txt`: Lists Python dependencies for the FastAPI application.

  - `docker-compose.yml`: Defines multi-container deployment for the web API services.

- src: Contains Python scripts used for various tasks such as data cleaning, manipulation, and utility functions.

  - `engineering_utils.py`: Contains functions related to feature engineering, including transformations and data preprocessing steps.
  - `stats_utils.py`: Contains utility functions for performing statistical analysis, including hypothesis testing and confidence interval calculations.
  - `utils.py`: General utility functions for data manipulation, cleaning, and visualizations.
  - `__init__.py`: Marks the directory as a Python package.

- stroke_prediction.ipynb: Jupyter notebook containing the full analysis, model building, and results for the stroke prediction project.

- .gitignore: Specifies the files and directories to ignore in version control (e.g., `.ipynb_checkpoints`, `__pycache__`).

- README.md: Provides an overview of the project, its goals, and how to use it.

- requirements.txt: Lists all required Python dependencies for the project, such as `pandas`, `numpy`, `matplotlib`, `scikit-learn`, etc.

## Technologies Used

- **Python 3.x**: The primary programming language used for analysis.
- **pandas**: For data manipulation and cleaning.
- **matplotlib**: For generating plots and visualizations.
- **seaborn**: For creating statistical visualizations like boxplots and correlation heatmaps.
- **scikit-learn**: For scaling data, preprocessing, and building machine learning models.
- **statsmodels**: For statistical analysis and fitting OLS models.
- **folium**: For creating interactive maps (if needed in future use cases).
- **shapely**: For geometric operations on map data (if needed).
- **scipy**: For statistical analysis and hypothesis testing.
- **ast**: For parsing literal expressions in strings.

## How to Use

### Prerequisites

- Python 3.8 or later
- `pip` for managing Python packages

### Installation

1. Clone the repository:

   ```bash
    git clone <repository-url>
    cd <module-folder>

    python -m venv venv
    venv\Scripts\activate

    pip install -r requirements.txt
   ```
