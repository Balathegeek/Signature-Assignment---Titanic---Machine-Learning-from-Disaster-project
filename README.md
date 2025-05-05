
# Titanic Survival Prediction Project

## Project Overview
This project aims to predict the survival of passengers aboard the RMS Titanic using machine learning techniques. The dataset includes various features such as age, gender, passenger class, fare, and port of embarkation. The goal is to develop a robust model that accurately predicts survival outcomes and provides insights into the factors influencing survival.

## Dataset Information
The dataset used in this project is the Titanic dataset from the Kaggle competition "Titanic: Machine Learning from Disaster." It contains information on 891 passengers, including:
- PassengerId: Unique identifier
- Survived: Target variable (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name, Sex, Age: Personal attributes
- SibSp, Parch: Number of siblings/spouses and parents/children aboard
- Ticket, Fare: Ticket number and fare paid
- Cabin: Cabin number (many missing values)
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## CRISP-DM Process
The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology:
1. **Business Understanding**: Define the problem and project objectives.
2. **Data Understanding**: Explore the dataset, identify data quality issues, and understand data distributions.
3. **Data Preparation**: Clean the data, handle missing values, and perform feature engineering.
4. **Modeling**: Train multiple machine learning models and evaluate their performance.
5. **Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
6. **Deployment**: Deploy the best-performing model and provide insights.

## Repository Structure
The repository is organized as follows:
- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data exploration, preprocessing, modeling, and evaluation.
- `scripts/`: Python scripts for data preprocessing, feature engineering, and model training.
- `models/`: Saved models and evaluation results.
- `README.md`: Project documentation.

## How to Run the Project
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd <repository_directory>`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Jupyter notebooks in the `notebooks/` directory to explore the data, preprocess it, and train the models.
5. Execute the Python scripts in the `scripts/` directory for data preprocessing, feature engineering, and model training.

## Dependencies
The project requires the following dependencies:
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Results and Insights
The project aims to develop a machine learning model that accurately predicts Titanic passenger survival. Key findings include:
- **Feature Importance**: Sex, Pclass, Fare, and Title are the most influential features in predicting survival.
- **Model Performance**: Gradient Boosting achieved the highest accuracy and ROC-AUC, making it the best-performing model.
- **Impact of Feature Engineering**: Interaction terms and polynomial features significantly improved model performance.
- **Class Imbalance Handling**: Applying SMOTE improved recall and F1-score, especially for the minority class (survivors).

By following the CRISP-DM process and leveraging advanced feature engineering techniques, the project demonstrates the importance of thoughtful data preparation and modeling strategies in solving real-world classification problems.
