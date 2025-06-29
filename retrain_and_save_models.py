import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load data
df = pd.read_csv('bank.csv')

# Apply the same filters as in the app
dff = df.copy()
dff = dff[dff['age'] < 75]
dff = dff.loc[dff["balance"] > -2257]
dff = dff.loc[dff["balance"] < 4087]
dff = dff.loc[dff["campaign"] < 6]
dff = dff.loc[dff["previous"] < 2.5]
bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff['Client_Category_M'] = pd.cut(dff['pdays'], bins=bins, labels=labels)
dff['Client_Category_M'] = dff['Client_Category_M'].astype('object')
liste_annee = []
for i in dff["month"]:
    if i in ["jun", "jul", "aug", "sep", "oct", "nov", "dec"]:
        liste_annee.append("2013")
    else:
        liste_annee.append("2014")
dff["year"] = liste_annee
dff['date'] = dff['day'].astype(str) + '-' + dff['month'].astype(str) + '-' + dff['year'].astype(str)
dff['date'] = pd.to_datetime(dff['date'])
dff["weekday"] = dff["date"].dt.weekday
dic = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
dff["weekday"] = dff["weekday"].replace(dic)

# Drop columns as in the app
dff = dff.drop(['contact', 'pdays', 'day', 'date', 'year'], axis=1)
dff['job'] = dff['job'].replace('unknown', np.nan)
dff['education'] = dff['education'].replace('unknown', np.nan)
dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

X = dff.drop('deposit', axis=1)
y = dff['deposit'].map({'yes': 1, 'no': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

# Impute and encode as in the app
imputer = SimpleImputer(strategy='most_frequent')
X_train.loc[:, ['job']] = imputer.fit_transform(X_train[['job']])
X_test.loc[:, ['job']] = imputer.transform(X_test[['job']])
X_train['poutcome'] = X_train['poutcome'].bfill().fillna(X_train['poutcome'].mode()[0])
X_test['poutcome'] = X_test['poutcome'].bfill().fillna(X_test['poutcome'].mode()[0])
X_train['education'] = X_train['education'].bfill().fillna(X_train['education'].mode()[0])
X_test['education'] = X_test['education'].bfill().fillna(X_test['education'].mode()[0])

# Standardize
scaler = StandardScaler()
cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
X_test[cols_num] = scaler.transform(X_test[cols_num])

# Encode binary categorical
oneh = OneHotEncoder(drop='first', sparse_output=False)
cat1 = ['default', 'housing', 'loan']
X_train[cat1] = oneh.fit_transform(X_train[cat1])
X_test[cat1] = oneh.transform(X_test[cat1])
X_train[cat1] = X_train[cat1].astype('int64')
X_test[cat1] = X_test[cat1].astype('int64')

# Ordinal encoding
X_train['education'] = X_train['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test['education'] = X_test['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_train['Client_Category_M'] = X_train['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test['Client_Category_M'] = X_test['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])

# Get dummies for multi-category
for col in ['job', 'marital', 'poutcome', 'month', 'weekday']:
    dummies = pd.get_dummies(X_train[col], prefix=col).astype(int)
    X_train = pd.concat([X_train.drop(col, axis=1), dummies], axis=1)
    dummies = pd.get_dummies(X_test[col], prefix=col).astype(int)
    X_test = pd.concat([X_test.drop(col, axis=1), dummies], axis=1)

# Train and save all major models
models = {
    'Random_Forest_model_avec_duration_sans_parametres.pkl': RandomForestClassifier(random_state=42),
    'XGBOOST_model_avec_duration_sans_parametres.pkl': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM_model_avec_duration_sans_parametres.pkl': SVC(probability=True, random_state=42),
    'KNN_model_avec_duration_sans_parametres.pkl': KNeighborsClassifier(),
    'Logistic_Regression_model_avec_duration_sans_parametres.pkl': LogisticRegression(max_iter=1000, random_state=42),
    'Decision_Tree_model_avec_duration_sans_parametres.pkl': DecisionTreeClassifier(random_state=42),
    'AdaBoost_model_avec_duration_sans_parametres.pkl': AdaBoostClassifier(random_state=42),
    'Bagging_model_avec_duration_sans_parametres.pkl': BaggingClassifier(random_state=42),
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, filename)
    print(f"Saved {filename}")

print('All avec_duration models retrained and saved successfully.')

# TEAM models with specific hyperparameters
team_models = {
    'RF_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl': RandomForestClassifier(max_depth=None, max_features='log2', min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42),
    'RF_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl': RandomForestClassifier(max_depth=None, max_features='log2', min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42),
    'RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl': RandomForestClassifier(class_weight='balanced', max_depth=20, max_features='sqrt', min_samples_leaf=2, min_samples_split=10, n_estimators=200, random_state=42),
    'SVM_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl': SVC(C=1, class_weight='balanced', gamma='scale', kernel='rbf', probability=True, random_state=42),
    'SVM_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl': SVC(C=0.1, class_weight='balanced', gamma=0.1, kernel='rbf', probability=True, random_state=42),
    'SVM_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl': SVC(kernel='rbf', gamma='scale', C=1, probability=True, random_state=42),
    'SVM_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl': SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='rbf', probability=True, random_state=42),
    'XGBOOST_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl': XGBClassifier(colsample_bytree=1.0, learning_rate=0.05, max_depth=7, min_child_weight=1, n_estimators=200, subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'XGBOOST_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl': XGBClassifier(base_score=0.3, gamma=14, learning_rate=0.6, max_delta_step=1, max_depth=27, min_child_weight=2, n_estimators=900, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'XGBOOST_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl': XGBClassifier(colsample_bytree=0.8, gamma=10, max_depth=17, min_child_weight=1, n_estimators=1000, reg_lambda=0.89, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'XGBOOST_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl': XGBClassifier(colsample_bytree=0.8, gamma=5, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'),
}

for filename, model in team_models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, filename)
    print(f"Saved {filename}")

# GridSearch2 models
grid2_models = {
    'Random_Forest_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl': RandomForestClassifier(class_weight='balanced', max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=15, n_estimators=200, random_state=42),
    'SVM_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl': SVC(C=1, class_weight='balanced', gamma='scale', kernel='rbf', probability=True, random_state=42),
    'XGBOOST_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl': XGBClassifier(colsample_bytree=0.8, gamma=5, learning_rate=0.05, max_depth=17, min_child_weight=1, n_estimators=200, subsample=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss'),
}

for filename, model in grid2_models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, filename)
    print(f"Saved {filename}")

print('All TEAM and GridSearch2 models retrained and saved successfully.') 