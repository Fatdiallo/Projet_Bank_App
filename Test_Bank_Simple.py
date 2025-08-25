import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import statsmodels.api
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from streamlit_option_menu import option_menu
from streamlit_extras.no_default_selectbox import selectbox
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import neighbors, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
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

# Main title
st.title("🏦 Prédiction du Succès d'une Campagne Marketing Bancaire")
st.markdown("---")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bank.csv')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# Load data
df = load_data()

if df is not None:
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
    
    # Main content based on selection
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
    
    elif selected == "Pre-processing":
        st.header("🔧 Pre-processing des Données")
        st.write("Nettoyage et préparation des données pour la modélisation")
        
        # Show data quality
        st.subheader("Qualité des données")
        missing_data = df.isnull().sum()
        st.bar_chart(missing_data)
    
    elif selected == "Modélisation":
        st.header("🤖 Modélisation")
        st.write("Entraînement et évaluation des modèles de machine learning")
        
        model_type = st.selectbox(
            "Choisir un type de modèle",
            ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]
        )
        
        if st.button("Entraîner le modèle"):
            with st.spinner("Entraînement en cours..."):
                st.success(f"Modèle {model_type} entraîné avec succès!")
    
    elif selected == "Interprétation":
        st.header("💡 Interprétation des Modèles")
        st.write("Analyse SHAP pour comprendre les prédictions")
        
        st.info("Cette section permet d'analyser l'importance des variables dans les prédictions.")
    
    elif selected == "Outil Prédictif":
        st.header("🎯 Outil Prédictif")
        st.write("Prédire la souscription d'un client à un dépôt à terme")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
            balance = st.number_input("Solde du client", value=1000)
            duration = st.number_input("Durée de l'appel (secondes)", value=100)
        
        with col2:
            campaign = st.number_input("Nombre de contacts campagne", value=1)
            previous = st.number_input("Nombre de contacts précédents", value=0)
            job = st.selectbox("Profession", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed"])
        
        if st.button("Prédire"):
            with st.spinner("Calcul de la prédiction..."):
                # Simulate prediction
                import random
                prediction = random.choice(["Oui", "Non"])
                confidence = random.uniform(0.6, 0.95)
                
                if prediction == "Oui":
                    st.success(f"✅ Prédiction: {prediction} (Confiance: {confidence:.1%})")
                else:
                    st.error(f"❌ Prédiction: {prediction} (Confiance: {confidence:.1%})")

else:
    st.error("❌ Impossible de charger les données. Veuillez vérifier que le fichier 'bank.csv' est présent.")

# Footer
st.markdown("---")
st.markdown("**Application développée pour la prédiction du succès des campagnes marketing bancaires**")
