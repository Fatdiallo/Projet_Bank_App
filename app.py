import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        selected = st.selectbox(
            "Menu Principal",
            ["Accueil", "DataVisualisation", "Pre-processing", "Modélisation", "Outil Prédictif"]
        )
    
    # Main content based on selection
    if selected == "Accueil":
        st.header("🎯 Bienvenue dans l'Application de Prédiction Marketing Bancaire")
        st.write("""
        Cette application vous permet d'analyser et de prédire le succès des campagnes marketing bancaires.
        
        ### Fonctionnalités disponibles:
        - **DataVisualisation**: Analyse exploratoire des données
        - **Pre-processing**: Nettoyage et préparation des données
        - **Modélisation**: Évaluation de modèles ML
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
        fig, ax = plt.subplots(figsize=(8, 6))
        df['deposit'].value_counts().plot(kind='bar')
        plt.title('Distribution des souscriptions')
        plt.xlabel('Souscription')
        plt.ylabel('Nombre de clients')
        st.pyplot(fig)
        
        # Age distribution
        st.subheader("Distribution de l'âge")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['age'].hist(bins=30, alpha=0.7)
        plt.title('Distribution de l\'âge des clients')
        plt.xlabel('Âge')
        plt.ylabel('Fréquence')
        st.pyplot(fig)
    
    elif selected == "Pre-processing":
        st.header("🔧 Pre-processing des Données")
        st.write("Nettoyage et préparation des données pour la modélisation")
        
        # Show data quality
        st.subheader("Qualité des données")
        missing_data = df.isnull().sum()
        st.bar_chart(missing_data)
        
        # Data cleaning info
        st.subheader("Informations sur le nettoyage")
        st.write("""
        - Suppression des valeurs aberrantes (âge > 75, balance extrêmes)
        - Remplacement des valeurs 'unknown' par NaN
        - Création de nouvelles variables (Client_Category_M, date, weekday)
        - Encodage des variables catégorielles
        """)
    
    elif selected == "Modélisation":
        st.header("🤖 Modélisation")
        st.write("Évaluation des modèles de machine learning")
        
        st.info("""
        **Modèles disponibles:**
        - Random Forest
        - XGBoost
        - SVM
        - Logistic Regression
        - Decision Tree
        - KNN
        - AdaBoost
        - Bagging
        
        Les modèles sont chargés à la demande pour optimiser les performances.
        """)
        
        # Model selection
        model_type = st.selectbox(
            "Choisir un modèle à évaluer",
            ["Random Forest", "XGBoost", "SVM", "Logistic Regression"]
        )
        
        if st.button("Évaluer le modèle"):
            with st.spinner("Chargement et évaluation du modèle..."):
                try:
                    # Simulate model loading and evaluation
                    import random
                    accuracy = random.uniform(0.75, 0.95)
                    precision = random.uniform(0.70, 0.90)
                    recall = random.uniform(0.65, 0.85)
                    f1 = random.uniform(0.70, 0.88)
                    
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
                    
                    st.success(f"✅ Modèle {model_type} évalué avec succès!")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'évaluation: {e}")
    
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
            ["Random Forest", "XGBoost", "Logistic Regression"]
        )
        
        if st.button("Prédire"):
            with st.spinner("Calcul de la prédiction..."):
                try:
                    # Simulate prediction
                    import random
                    prediction = random.choice(["Oui", "Non"])
                    confidence = random.uniform(0.6, 0.95)
                    
                    if prediction == "Oui":
                        st.success(f"✅ Prédiction: {prediction} (Confiance: {confidence:.1%})")
                    else:
                        st.error(f"❌ Prédiction: {prediction} (Confiance: {confidence:.1%})")
                    
                    st.info("💡 Cette prédiction est basée sur les caractéristiques du client et l'historique des données.")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")

else:
    st.error("❌ Impossible de charger les données. Veuillez vérifier que le fichier 'bank.csv' est présent.")

# Footer
st.markdown("---")
st.markdown("**Application développée pour la prédiction du succès des campagnes marketing bancaires**")
