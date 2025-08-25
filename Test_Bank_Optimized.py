import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
 
import scipy.stats as stats

import statsmodels.api
import sklearn
print(sklearn.__version__)


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from streamlit_option_menu import option_menu
from streamlit_extras.no_default_selectbox import selectbox

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn import neighbors
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report

import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Utility function to align test data to model's expected columns
def align_X_test(X_test, model):
    """Align X_test columns to match model's expected feature names"""
    if hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in X_test.columns:
                X_test[col] = 0
        return X_test[expected_cols]
    else:
        return X_test

# Load data with caching
@st.cache_data
def load_and_process_data():
    """Load and process the bank data"""
    try:
        df = pd.read_csv('bank.csv')
        
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

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=48)
        
        # Process data (same as original)
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_train.loc[:, ['job']] = imputer.fit_transform(X_train[['job']])
        X_test.loc[:, ['job']] = imputer.transform(X_test[['job']])

        # Fill NaN values
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

        # One-hot encode other categorical variables
        categorical_vars = ['job', 'marital', 'poutcome', 'month', 'weekday']
        for var in categorical_vars:
            dummies_train = pd.get_dummies(X_train[var], prefix=var).astype(int)
            dummies_test = pd.get_dummies(X_test[var], prefix=var).astype(int)
            X_train = pd.concat([X_train.drop(var, axis=1), dummies_train], axis=1)
            X_test = pd.concat([X_test.drop(var, axis=1), dummies_test], axis=1)

        return X_train, X_test, y_train, y_test, scaler, le, df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None, None, None, None, None, None

# Load models on-demand
@st.cache_resource
def load_model(model_name):
    """Load a specific model on demand"""
    try:
        model_files = {
            "Random Forest": "Random_Forest_model_sans_duration_sans_parametres.pkl",
            "Logistic Regression": "Logistic_Regression_model_sans_duration_sans_parametres.pkl",
            "Decision Tree": "Decision_Tree_model_sans_duration_sans_parametres.pkl",
            "KNN": "KNN_model_sans_duration_sans_parametres.pkl",
            "AdaBoost": "AdaBoost_model_sans_duration_sans_parametres.pkl",
            "Bagging": "Bagging_model_sans_duration_sans_parametres.pkl",
            "SVM": "SVM_model_sans_duration_sans_parametres.pkl",
            "XGBOOST": "XGBOOST_model_sans_duration_sans_parametres.pkl"
        }
        
        if model_name in model_files:
            file_path = model_files[model_name]
            if os.path.exists(file_path):
                return joblib.load(file_path)
            else:
                st.error(f"Fichier modèle {file_path} non trouvé")
                return None
        else:
            st.error(f"Modèle {model_name} non reconnu")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
        return None

# Main app
def main():
    st.title("🏦 Prédiction du Succès d'une Campagne Marketing Bancaire")
    st.markdown("---")
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, le, df = load_and_process_data()
    
    if X_train is not None:
        st.success(f"✅ Données chargées avec succès! {df.shape[0]} clients, {df.shape[1]} variables")
        
        # Sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Menu Principal",
                options=["Accueil", "DataVisualisation", "Pre-processing", "Modélisation", "Interprétation", "Outil Prédictif"],
                icons=["house", "graph-up", "gear", "cpu", "lightbulb", "calculator"],
                menu_icon="cast",
                default_index=0,
            )
        
        # Main content
        if selected == "Accueil":
            st.header("🎯 Bienvenue dans l'Application de Prédiction Marketing Bancaire")
            st.write("""
            Cette application vous permet d'analyser et de prédire le succès des campagnes marketing bancaires.
            
            ### Fonctionnalités disponibles:
            - **DataVisualisation**: Analyse exploratoire des données
            - **Pre-processing**: Nettoyage et préparation des données
            - **Modélisation**: Entraînement et évaluation de modèles ML
            - **Interprétation**: Analyse SHAP pour l'explicabilité
            - **Outil Prédictif**: Interface de prédiction interactive
            """)
            
            # Display basic dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre de clients", df.shape[0])
            with col2:
                st.metric("Nombre de variables", df.shape[1])
            with col3:
                deposit_rate = (df['deposit'] == 'yes').mean() * 100
                st.metric("Taux de souscription", f"{deposit_rate:.1f}%")
        
        elif selected == "DataVisualisation":
            st.header("📊 Visualisation des Données")
            st.write("Analyse exploratoire des données de marketing bancaire")
            
            # Basic statistics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Statistiques descriptives")
                st.dataframe(df.describe())
            
            with col2:
                st.subheader("Types de données")
                st.dataframe(df.dtypes.to_frame('Type'))
            
            # Distribution of target variable
            st.subheader("Distribution de la variable cible")
            fig, ax = plt.subplots()
            df['deposit'].value_counts().plot(kind='bar')
            plt.title('Distribution des souscriptions')
            plt.xlabel('Souscription')
            plt.ylabel('Nombre de clients')
            st.pyplot(fig)
        
        elif selected == "Pre-processing":
            st.header("🔧 Pre-processing des Données")
            st.write("Nettoyage et préparation des données pour la modélisation")
            
            # Show data quality
            st.subheader("Qualité des données")
            missing_data = df.isnull().sum()
            st.bar_chart(missing_data)
            
            st.subheader("Données après nettoyage")
            st.write(f"Forme des données d'entraînement: {X_train.shape}")
            st.write(f"Forme des données de test: {X_test.shape}")
        
        elif selected == "Modélisation":
            st.header("🤖 Modélisation")
            st.write("Évaluation des modèles de machine learning")
            
            model_type = st.selectbox(
                "Choisir un modèle à évaluer",
                ["Random Forest", "Logistic Regression", "Decision Tree", "KNN", "AdaBoost", "Bagging", "SVM", "XGBOOST"]
            )
            
            if st.button("Évaluer le modèle"):
                with st.spinner("Chargement et évaluation du modèle..."):
                    model = load_model(model_type)
                    if model is not None:
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.3f}")
                        with col2:
                            st.metric("Precision", f"{precision:.3f}")
                        with col3:
                            st.metric("Recall", f"{recall:.3f}")
                        with col4:
                            st.metric("F1-Score", f"{f1:.3f}")
                        
                        # Confusion matrix
                        st.subheader("Matrice de confusion")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Matrice de confusion - {model_type}')
                        plt.ylabel('Vraie classe')
                        plt.xlabel('Classe prédite')
                        st.pyplot(fig)
        
        elif selected == "Interprétation":
            st.header("💡 Interprétation des Modèles")
            st.write("Analyse SHAP pour comprendre les prédictions")
            
            st.info("Cette section permet d'analyser l'importance des variables dans les prédictions.")
            st.write("Pour une analyse SHAP complète, veuillez utiliser l'application complète.")
        
        elif selected == "Outil Prédictif":
            st.header("🎯 Outil Prédictif")
            st.write("Prédire la souscription d'un client à un dépôt à terme")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
                balance = st.number_input("Solde du client", value=1000)
                duration = st.number_input("Durée de l'appel (secondes)", value=100)
                campaign = st.number_input("Nombre de contacts campagne", value=1)
            
            with col2:
                previous = st.number_input("Nombre de contacts précédents", value=0)
                job = st.selectbox("Profession", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"])
                education = st.selectbox("Éducation", ["primary", "secondary", "tertiary"])
                marital = st.selectbox("Statut marital", ["single", "married", "divorced"])
            
            model_choice = st.selectbox(
                "Choisir le modèle pour la prédiction",
                ["Random Forest", "XGBOOST", "Logistic Regression"]
            )
            
            if st.button("Prédire"):
                with st.spinner("Calcul de la prédiction..."):
                    model = load_model(model_choice)
                    if model is not None:
                        # Create sample data (simplified)
                        st.success(f"✅ Modèle {model_choice} chargé avec succès!")
                        st.info("Pour une prédiction complète, veuillez utiliser l'application complète avec tous les modèles.")
                    else:
                        st.error("❌ Erreur lors du chargement du modèle")

    else:
        st.error("❌ Impossible de charger les données. Veuillez vérifier que le fichier 'bank.csv' est présent.")

    # Footer
    st.markdown("---")
    st.markdown("**Application optimisée pour la prédiction du succès des campagnes marketing bancaires**")

if __name__ == "__main__":
    main()
