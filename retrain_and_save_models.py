import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv('bank.csv')

def train_and_save_models(X, y, suffix):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numerical_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        f'Random_Forest_model_{suffix}.pkl': RandomForestClassifier(random_state=42),
        f'XGBOOST_model_{suffix}.pkl': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        f'SVM_model_{suffix}.pkl': SVC(probability=True, random_state=42),
        f'KNN_model_{suffix}.pkl': KNeighborsClassifier(),
        f'Logistic_Regression_model_{suffix}.pkl': LogisticRegression(max_iter=1000, random_state=42),
        f'Decision_Tree_model_{suffix}.pkl': DecisionTreeClassifier(random_state=42),
        f'AdaBoost_model_{suffix}.pkl': AdaBoostClassifier(random_state=42),
        f'Bagging_model_{suffix}.pkl': BaggingClassifier(random_state=42),
    }

    for filename, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, filename)
        print(f"Saved {filename}")

# Prepare target
y = df['deposit'].map({'yes': 1, 'no': 0})

# With duration
X_avec = df.drop('deposit', axis=1)
train_and_save_models(X_avec, y, 'avec_duration_sans_parametres')

# Without duration
X_sans = df.drop(['deposit', 'duration'], axis=1)
train_and_save_models(X_sans, y, 'sans_duration_sans_parametres')

print('All models retrained and saved successfully.') 