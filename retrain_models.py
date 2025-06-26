import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data (same as in main script)
df = pd.read_csv('bank.csv')

# Data preprocessing for WITH duration
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
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec":
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i == "apr" or i == "may":
        liste_annee.append("2014")
dff["year"] = liste_annee
dff['date'] = dff['day'].astype(str) + '-' + dff['month'].astype(str) + '-' + dff['year'].astype(str)
dff['date'] = pd.to_datetime(dff['date'])
dff["weekday"] = dff["date"].dt.weekday
dic = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
dff["weekday"] = dff["weekday"].replace(dic)

dff = dff.drop(['contact'], axis=1)
dff = dff.drop(['pdays'], axis=1)
dff = dff.drop(['day'], axis=1)
dff = dff.drop(['date'], axis=1)
dff = dff.drop(['year'], axis=1)
dff['job'] = dff['job'].replace('unknown', np.nan)
dff['education'] = dff['education'].replace('unknown', np.nan)
dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

X = dff.drop('deposit', axis=1)
y = dff['deposit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=48)

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train.loc[:, ['job']] = imputer.fit_transform(X_train[['job']])
X_test.loc[:, ['job']] = imputer.transform(X_test[['job']])

X_train['poutcome'] = X_train['poutcome'].fillna(method='bfill')
X_train['poutcome'] = X_train['poutcome'].fillna(X_train['poutcome'].mode()[0])
X_test['poutcome'] = X_test['poutcome'].fillna(method='bfill')
X_test['poutcome'] = X_test['poutcome'].fillna(X_test['poutcome'].mode()[0])

X_train['education'] = X_train['education'].fillna(method='bfill')
X_train['education'] = X_train['education'].fillna(X_train['education'].mode()[0])
X_test['education'] = X_test['education'].fillna(method='bfill')
X_test['education'] = X_test['education'].fillna(X_test['education'].mode()[0])

# Standardization
scaler = StandardScaler()
cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
X_test[cols_num] = scaler.transform(X_test[cols_num])

# Encode target
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Encode categorical variables
oneh = OneHotEncoder(drop='first', sparse_output=False)
cat1 = ['default', 'housing', 'loan']
X_train.loc[:, cat1] = oneh.fit_transform(X_train[cat1])
X_test.loc[:, cat1] = oneh.transform(X_test[cat1])

X_train[cat1] = X_train[cat1].astype('int64')
X_test[cat1] = X_test[cat1].astype('int64')

# Encode ordinal variables
X_train['education'] = X_train['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test['education'] = X_test['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

X_train['Client_Category_M'] = X_train['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test['Client_Category_M'] = X_test['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])

# One-hot encode multi-category variables
for col in ['job', 'marital', 'poutcome', 'month', 'weekday']:
    dummies = pd.get_dummies(X_train[col], prefix=col).astype(int)
    X_train = pd.concat([X_train.drop(col, axis=1), dummies], axis=1)
    dummies = pd.get_dummies(X_test[col], prefix=col).astype(int)
    X_test = pd.concat([X_test.drop(col, axis=1), dummies], axis=1)

# Data preprocessing for WITHOUT duration
dff_sans_duration = df.copy()
dff_sans_duration = dff_sans_duration[dff_sans_duration['age'] < 75]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] > -2257]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] < 4087]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["campaign"] < 6]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["previous"] < 2.5]
dff_sans_duration = dff_sans_duration.drop('contact', axis=1)

bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff_sans_duration['Client_Category_M'] = pd.cut(dff_sans_duration['pdays'], bins=bins, labels=labels)
dff_sans_duration['Client_Category_M'] = dff_sans_duration['Client_Category_M'].astype('object')
dff_sans_duration = dff_sans_duration.drop('pdays', axis=1)

liste_annee = []
for i in dff_sans_duration["month"]:
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec":
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i == "apr" or i == "may":
        liste_annee.append("2014")
dff_sans_duration["year"] = liste_annee
dff_sans_duration['date'] = dff_sans_duration['day'].astype(str) + '-' + dff_sans_duration['month'].astype(str) + '-' + dff_sans_duration['year'].astype(str)
dff_sans_duration['date'] = pd.to_datetime(dff_sans_duration['date'])
dff_sans_duration["weekday"] = dff_sans_duration["date"].dt.weekday
dic = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
dff_sans_duration["weekday"] = dff_sans_duration["weekday"].replace(dic)

dff_sans_duration = dff_sans_duration.drop(['day'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['date'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['year'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['duration'], axis=1)

dff_sans_duration['job'] = dff_sans_duration['job'].replace('unknown', np.nan)
dff_sans_duration['education'] = dff_sans_duration['education'].replace('unknown', np.nan)
dff_sans_duration['poutcome'] = dff_sans_duration['poutcome'].replace('unknown', np.nan)

X_sans_duration = dff_sans_duration.drop('deposit', axis=1)
y_sans_duration = dff_sans_duration['deposit']

# Train-test split for without duration
X_train_sd, X_test_sd, y_train_sd, y_test_sd = train_test_split(X_sans_duration, y_sans_duration, test_size=0.20, random_state=48)

# Handle missing values for without duration
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train_sd.loc[:, ['job']] = imputer.fit_transform(X_train_sd[['job']])
X_test_sd.loc[:, ['job']] = imputer.transform(X_test_sd[['job']])

X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(method='bfill')
X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(X_train_sd['poutcome'].mode()[0])
X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(method='bfill')
X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(X_test_sd['poutcome'].mode()[0])

X_train_sd['education'] = X_train_sd['education'].fillna(method='bfill')
X_train_sd['education'] = X_train_sd['education'].fillna(X_train_sd['education'].mode()[0])
X_test_sd['education'] = X_test_sd['education'].fillna(method='bfill')
X_test_sd['education'] = X_test_sd['education'].fillna(X_test_sd['education'].mode()[0])

# Standardization for without duration
scaler_sd = StandardScaler()
cols_num_sd = ['age', 'balance', 'campaign', 'previous']
X_train_sd[cols_num_sd] = scaler_sd.fit_transform(X_train_sd[cols_num_sd])
X_test_sd[cols_num_sd] = scaler_sd.transform(X_test_sd[cols_num_sd])

# Encode target for without duration
le_sd = LabelEncoder()
y_train_sd = le_sd.fit_transform(y_train_sd)
y_test_sd = le_sd.transform(y_test_sd)

# Encode categorical variables for without duration
oneh_sd = OneHotEncoder(drop='first', sparse_output=False)
cat1_sd = ['default', 'housing', 'loan']
X_train_sd.loc[:, cat1_sd] = oneh_sd.fit_transform(X_train_sd[cat1_sd])
X_test_sd.loc[:, cat1_sd] = oneh_sd.transform(X_test_sd[cat1_sd])

X_train_sd[cat1_sd] = X_train_sd[cat1_sd].astype('int64')
X_test_sd[cat1_sd] = X_test_sd[cat1_sd].astype('int64')

# Encode ordinal variables for without duration
X_train_sd['education'] = X_train_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test_sd['education'] = X_test_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

X_train_sd['Client_Category_M'] = X_train_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test_sd['Client_Category_M'] = X_test_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])

# One-hot encode multi-category variables for without duration
for col in ['job', 'marital', 'poutcome', 'month', 'weekday']:
    dummies_sd = pd.get_dummies(X_train_sd[col], prefix=col).astype(int)
    X_train_sd = pd.concat([X_train_sd.drop(col, axis=1), dummies_sd], axis=1)
    dummies_sd = pd.get_dummies(X_test_sd[col], prefix=col).astype(int)
    X_test_sd = pd.concat([X_test_sd.drop(col, axis=1), dummies_sd], axis=1)

print("Data preprocessing completed. Now retraining models...")

# Retrain the problematic models for WITHOUT duration
print("Retraining models for WITHOUT duration...")

# Random Forest model
rf_model = RandomForestClassifier(class_weight='balanced', max_depth=8, max_features='log2', 
                                 min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42)
rf_model.fit(X_train_sd, y_train_sd)
joblib.dump(rf_model, "Random_Forest_model_SD_TOP_4_hyperparam.pkl")
print("Random Forest model retrained and saved")

# Decision Tree model
dt_model = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5, 
                                 max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42)
dt_model.fit(X_train_sd, y_train_sd)
joblib.dump(dt_model, "Decision_Tree_model_SD_TOP_4_hyperparam.pkl")
print("Decision Tree model retrained and saved")

# SVM model
svm_model = SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear', random_state=42)
svm_model.fit(X_train_sd, y_train_sd)
joblib.dump(svm_model, "SVM_model_SD_TOP_4_hyperparam.pkl")
print("SVM model retrained and saved")

# XGBOOST model
xgb_model = XGBClassifier(gamma=0.05, colsample_bytree=0.9, learning_rate=0.39, max_depth=6, 
                         min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, 
                         scale_pos_weight=2.6, subsample=0.99, random_state=42)
xgb_model.fit(X_train_sd, y_train_sd)
joblib.dump(xgb_model, "XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
print("XGBOOST model retrained and saved")

print("All models retrained successfully with current scikit-learn version!") 