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

df=pd.read_csv('bank.csv')

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
liste_annee =[]
for i in dff["month"] :
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
        liste_annee.append("2014")
dff["year"] = liste_annee
dff['date'] = dff['day'].astype(str)+ '-'+ dff['month'].astype(str)+ '-'+ dff['year'].astype(str)
dff['date']= pd.to_datetime(dff['date'])
dff["weekday"] = dff["date"].dt.weekday
dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
dff["weekday"] = dff["weekday"].replace(dic)

dff = dff.drop(['contact'], axis=1)
dff = dff.drop(['pdays'], axis=1)
dff = dff.drop(['day'], axis=1)
dff = dff.drop(['date'], axis=1)
dff = dff.drop(['year'], axis=1)
dff['job'] = dff['job'].replace('unknown', np.nan)
dff['education'] = dff['education'].replace('unknown', np.nan)
dff['poutcome'] = dff['poutcome'].replace('unknown', np.nan)

X = dff.drop('deposit', axis = 1)
y = dff['deposit']

# Séparation des données en un jeu d'entrainement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 48)
                        
# Remplacement des NaNs par le mode:
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train.loc[:,['job']] = imputer.fit_transform(X_train[['job']])
X_test.loc[:,['job']] = imputer.transform(X_test[['job']])

# On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
# On l'applique au X_train et X_test :
X_train['poutcome'] = X_train['poutcome'].fillna(method ='bfill')
X_train['poutcome'] = X_train['poutcome'].fillna(X_train['poutcome'].mode()[0])

X_test['poutcome'] = X_test['poutcome'].fillna(method ='bfill')
X_test['poutcome'] = X_test['poutcome'].fillna(X_test['poutcome'].mode()[0])

# On fait de même pour les NaaN de 'education'
X_train['education'] = X_train['education'].fillna(method ='bfill')
X_train['education'] = X_train['education'].fillna(X_train['education'].mode()[0])

X_test['education'] = X_test['education'].fillna(method ='bfill')
X_test['education'] = X_test['education'].fillna(X_test['education'].mode()[0])
                        
# Standardisation des variables quantitatives:
scaler = StandardScaler()
cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
X_train [cols_num] = scaler.fit_transform(X_train [cols_num])
X_test [cols_num] = scaler.transform (X_test [cols_num])

# Encodage de la variable Cible 'deposit':
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Encodage des variables explicatives de type 'objet'
oneh = OneHotEncoder(drop = 'first', sparse_output = False)
cat1 = ['default', 'housing','loan']
X_train.loc[:, cat1] = oneh.fit_transform(X_train[cat1])
X_test.loc[:, cat1] = oneh.transform(X_test[cat1])

X_train[cat1] = X_train[cat1].astype('int64')
X_test[cat1] = X_test[cat1].astype('int64')

# 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train['education'] = X_train['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test['education'] = X_test['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

# 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train['Client_Category_M'] = X_train['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test['Client_Category_M'] = X_test['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


# Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
dummies = pd.get_dummies(X_train['job'], prefix='job').astype(int)
X_train = pd.concat([X_train.drop('job', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['job'], prefix='job').astype(int)
X_test = pd.concat([X_test.drop('job', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['marital'], prefix='marital').astype(int)
X_train = pd.concat([X_train.drop('marital', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['marital'], prefix='marital').astype(int)
X_test = pd.concat([X_test.drop('marital', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['poutcome'], prefix='poutcome').astype(int)
X_train = pd.concat([X_train.drop('poutcome', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['poutcome'], prefix='poutcome').astype(int)
X_test = pd.concat([X_test.drop('poutcome', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['month'], prefix='month').astype(int)
X_train = pd.concat([X_train.drop('month', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['month'], prefix='month').astype(int)
X_test = pd.concat([X_test.drop('month', axis=1), dummies], axis=1)

dummies = pd.get_dummies(X_train['weekday'], prefix='weekday').astype(int)
X_train = pd.concat([X_train.drop('weekday', axis=1), dummies], axis=1)
dummies = pd.get_dummies(X_test['weekday'], prefix='weekday').astype(int)
X_test = pd.concat([X_test.drop('weekday', axis=1), dummies], axis=1)

#Récupération des valeurs originales à partir des données standardisées
X_train_original = X_train.copy()
X_test_original = X_test.copy()

#Inversion de la standardisation
X_train_original[cols_num] = scaler.inverse_transform(X_train[cols_num])
X_test_original[cols_num] = scaler.inverse_transform(X_test[cols_num])

#code python SANS DURATION
dff_sans_duration = df.copy()
dff_sans_duration = dff_sans_duration[dff_sans_duration['age'] < 75]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] > -2257]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["balance"] < 4087]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["campaign"] < 6]
dff_sans_duration = dff_sans_duration.loc[dff_sans_duration["previous"] < 2.5]
dff_sans_duration = dff_sans_duration.drop('contact', axis = 1)

bins = [-2, -1, 180, 855]
labels = ['Prospect', 'Reached-6M', 'Reached+6M']
dff_sans_duration['Client_Category_M'] = pd.cut(dff_sans_duration['pdays'], bins=bins, labels=labels)
dff_sans_duration['Client_Category_M'] = dff_sans_duration['Client_Category_M'].astype('object')
dff_sans_duration = dff_sans_duration.drop('pdays', axis = 1)

liste_annee =[]
for i in dff_sans_duration["month"] :
    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
        liste_annee.append("2013")
    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
        liste_annee.append("2014")
dff_sans_duration["year"] = liste_annee
dff_sans_duration['date'] = dff_sans_duration['day'].astype(str)+ '-'+ dff_sans_duration['month'].astype(str)+ '-'+ dff_sans_duration['year'].astype(str)
dff_sans_duration['date']= pd.to_datetime(dff_sans_duration['date'])
dff_sans_duration["weekday"] = dff_sans_duration["date"].dt.weekday
dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
dff_sans_duration["weekday"] = dff_sans_duration["weekday"].replace(dic)

dff_sans_duration = dff_sans_duration.drop(['day'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['date'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['year'], axis=1)
dff_sans_duration = dff_sans_duration.drop(['duration'], axis=1)

dff_sans_duration['job'] = dff_sans_duration['job'].replace('unknown', np.nan)
dff_sans_duration['education'] = dff_sans_duration['education'].replace('unknown', np.nan)
dff_sans_duration['poutcome'] = dff_sans_duration['poutcome'].replace('unknown', np.nan)

X_sans_duration = dff_sans_duration.drop('deposit', axis = 1)
y_sans_duration = dff_sans_duration['deposit']

# Séparation des données en un jeu d'entrainement et jeu de test
X_train_sd, X_test_sd, y_train_sd, y_test_sd = train_test_split(X_sans_duration, y_sans_duration, test_size = 0.20, random_state = 48)
                
# Remplacement des NaNs par le mode:
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train_sd.loc[:,['job']] = imputer.fit_transform(X_train_sd[['job']])
X_test_sd.loc[:,['job']] = imputer.transform(X_test_sd[['job']])
 
# On remplace les NaaN de 'poutcome' avec la méthode de remplissage par propagation où chaque valeur unknown est remplacée par la valeur de la ligne suivante (puis la dernière ligne par le Mode de cette variable).
# On l'applique au X_train et X_test :
X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(method ='bfill')
X_train_sd['poutcome'] = X_train_sd['poutcome'].fillna(X_train_sd['poutcome'].mode()[0])

X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(method ='bfill')
X_test_sd['poutcome'] = X_test_sd['poutcome'].fillna(X_test_sd['poutcome'].mode()[0])

# On fait de même pour les NaaN de 'education'
X_train_sd['education'] = X_train_sd['education'].fillna(method ='bfill')
X_train_sd['education'] = X_train_sd['education'].fillna(X_train_sd['education'].mode()[0])

X_test_sd['education'] = X_test_sd['education'].fillna(method ='bfill')
X_test_sd['education'] = X_test_sd['education'].fillna(X_test_sd['education'].mode()[0])
            
# Standardisation des variables quantitatives:
scaler_sd = StandardScaler()
cols_num_sd = ['age', 'balance', 'campaign', 'previous']
X_train_sd[cols_num_sd] = scaler_sd.fit_transform(X_train_sd[cols_num_sd])
X_test_sd[cols_num_sd] = scaler_sd.transform (X_test_sd[cols_num_sd])

# Encodage de la variable Cible 'deposit':
le_sd = LabelEncoder()
y_train_sd = le_sd.fit_transform(y_train_sd)
y_test_sd = le_sd.transform(y_test_sd)

# Encodage des variables explicatives de type 'objet'
oneh_sd = OneHotEncoder(drop = 'first', sparse_output = False)
cat1_sd = ['default', 'housing','loan']
X_train_sd.loc[:, cat1_sd] = oneh_sd.fit_transform(X_train_sd[cat1_sd])
X_test_sd.loc[:, cat1_sd] = oneh_sd.transform(X_test_sd[cat1_sd])

X_train_sd[cat1_sd] = X_train_sd[cat1_sd].astype('int64')
X_test_sd[cat1_sd] = X_test_sd[cat1_sd].astype('int64')

# 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train_sd['education'] = X_train_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
X_test_sd['education'] = X_test_sd['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

# 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
X_train_sd['Client_Category_M'] = X_train_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
X_test_sd['Client_Category_M'] = X_test_sd['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


# Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
dummies_sd = pd.get_dummies(X_train_sd['job'], prefix='job').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('job', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['job'], prefix='job').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('job', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['marital'], prefix='marital').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('marital', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['marital'], prefix='marital').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('marital', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['poutcome'], prefix='poutcome').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('poutcome', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['poutcome'], prefix='poutcome').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('poutcome', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['month'], prefix='month').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('month', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['month'], prefix='month').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('month', axis=1), dummies_sd], axis=1)

dummies_sd = pd.get_dummies(X_train_sd['weekday'], prefix='weekday').astype(int)
X_train_sd = pd.concat([X_train_sd.drop('weekday', axis=1), dummies_sd], axis=1)
dummies_sd = pd.get_dummies(X_test_sd['weekday'], prefix='weekday').astype(int)
X_test_sd = pd.concat([X_test_sd.drop('weekday', axis=1), dummies_sd], axis=1)

#Récupération des valeurs originales à partir des données standardisées
X_train_sd_original = X_train_sd.copy()
X_test_sd_original = X_test_sd.copy()

#Inversion de la standardisation
X_train_sd_original[cols_num_sd] = scaler_sd.inverse_transform(X_train_sd[cols_num_sd])
X_test_sd_original[cols_num_sd] = scaler_sd.inverse_transform(X_test_sd[cols_num_sd])

with st.sidebar:
    selected = option_menu(
        menu_title='Sections',
        options=['Introduction','DataVisualisation', "Pre-processing", "Modélisation", "Interprétation", "Recommandations & Perspectives", "Outil  Prédictif"]) 
#with st.sidebar:
    #st.markdown("---")  
    #st.subheader("Membres du projet")  
    #st.markdown("- **Dilène SANTOS**")
    #st.markdown("- **Carolle DEUMENI**")
    #st.markdown("- **Fatoumata DIALLO**")
    #st.markdown("- **Douniazed FILALI**")
 
if selected == 'Introduction':  
    st.title("Prédiction du succès d'une campagne Marketing pour une banque")
    st.subheader("Contexte du projet")
    st.write("Le projet vise à analyser des données marketing issues d'une banque qui a utilisé le télémarketing pour **promouvoir un produit financier appelé 'dépôt à terme'**. Ce produit nécessite que le client dépose une somme d'argent dans un compte dédié, sans possibilité de retrait avant une date déterminée. En retour, le client reçoit des intérêts à la fin de cette période. **L'objectif de cette analyse est d'examiner les informations personnelles des clients, comme l'âge, le statut matrimonial, le montant d'argent déposé, le nombre de contacts réalisés, etc., afin de comprendre les facteurs qui influencent la décision des clients de souscrire ou non à ce produit financier.**")
    

    st.write("#### Problématique : ")
    st.write("La principale problématique de ce projet est de **déterminer** les **facteurs qui influencent la probabilité qu'un client souscrive à un dépôt à terme à la suite d'une campagne de télémarketing.**")
    st.write("L'objectif est double :")
    st.write("- Identifier et analyser visuellement et statistiquement **les caractéristiques des clients** qui sont corrélées avec la souscription au 'dépôt à terme'.")
    st.write("- Utiliser des techniques de Machine Learning pour **prédire si un client va souscrire au 'dépôt à terme'.**")

    st.write("#### Les données : ")
    st.markdown("Le jeu de données comprend un total de **11 162 lignes** et **17 colonnes**.  \n\
    Ces colonnes fournissent 3 types d'informations :")
    st.write("") 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**I. Infos socio-démo:**  \n\
        1. age  \n\
        2. job  \n\
        3. marital  \n\
        4. education")
    with col2:
        st.markdown("**II. Infos situation bancaire:**  \n\
        5. default  \n\
        6. balance  \n\
        7. housing  \n\
        8. loan")
        st.write("") 

    with col3:
        st.markdown("**III. Infos campagnes marketing:** \n\
        9. contact  \n\
        10. day  \n\
        11. month  \n\
        12. duration  \n\
        13. campaign  \n\
        14. pdays  \n\
        15. previous  \n\
        16. poutcome")
        st.write("") 
        
    st.write("**Notre variable cible:**  \n\
    17. deposit")
    

    
    

    
    
   
if selected == 'DataVisualisation':      
    pages = st.sidebar.radio("", ["Analyse Univariée", "Analyse Multivariée", "Profiling"])

    if pages == "Analyse Univariée" :  # Analyse Univariée
        st.title("Analyse Univariée")

        # Liste des variables qualitatives et quantitatives
        quantitative_vars = ["age", "duration", "campaign", "balance", "pdays", "previous"]
        qualitative_vars = ["job", "marital", "education", "default", "housing", "loan", 
                            "contact", "poutcome", "deposit", "weekday", "month"]

        # Sélection du type de variable        
        analysis_type = st.radio(
            " ",
            ["**VARIABLES QUALITATIVES**", "**VARIABLES QUANTITATIVES**"],
            key="type_variable_selectbox", horizontal=True
        )

        # Affichage des variables en fonction du type choisi
        if analysis_type == "**VARIABLES QUALITATIVES**":
            selected_variable = st.radio(
                " ",
                qualitative_vars,
                key="qualitative_var_selectbox", horizontal=True
            )
            st.write("____________________________________")

            st.write(f"Analyse de la variable qualitative : **{selected_variable}**")

            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in df["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            df["year"] = liste_annee

            df['date'] = df['day'].astype(str)+ '-'+ df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df['date']= pd.to_datetime(df['date'])

            df["weekday"] = df["date"].dt.weekday
            dic = {0 : "Lundi",
            1 : "Mardi",
            2 : "Mercredi",
            3 : "Jeudi",
            4 : "Vendredi",
            5 : "Samedi",
            6 : "Dimanche"}
            df["weekday"] = df["weekday"].replace(dic)

            # Analyse spécifique pour les variables qualitatives
            st.write("### Distribution des catégories")

            # Calcul des pourcentages
            category_counts = df[selected_variable].value_counts()
            category_percentages = category_counts / category_counts.sum() * 100
            
            # Création du graphique avec barres horizontales
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.countplot(
                y=selected_variable,  # Passer `y` pour un graphique horizontal
                data=df,
                color='c',
                order=category_counts.index,
                ax=ax
            )
            ax.set_ylabel("")
            
            # Ajouter les annotations pourcentages sur les barres
            for i, count in enumerate(category_counts):
                percentage = category_percentages.iloc[i]
                ax.text(count + 0.5, i, f"{percentage: .0f}%", va="center", fontsize=7)  # `va="center"` pour centrer verticalement
            
            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
            st.write("Le graphique ci-dessus montre la proportion de chaque catégorie dans la variable.")


        elif analysis_type == "**VARIABLES QUANTITATIVES**":
            selected_variable = st.radio(
                " ",
                quantitative_vars,
                key="quantitative_var_selectbox", horizontal=True
            )
            st.write("____________________________________")

            st.write(f"Analyse de la variable quantitative : **{selected_variable}**")

            # 1. Histogramme avec KDE
            # Créer un bouton
            st.write("### Distribution (Histogramme et KDE)") 
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[selected_variable], bins=20, kde=True, color='b', ax=ax)
            ax.set_title(f'Distribution de {selected_variable}', fontsize=14)
            ax.set_xlabel(selected_variable, fontsize=12)
            ax.set_ylabel('Fréquence', fontsize=12)
            st.pyplot(fig)

            # 2. Boxplot
            st.write("### Box Plot") 
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.boxplot(df[selected_variable], vert=False, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue'))
            ax.set_title(f'Box Plot de {selected_variable}', fontsize=14)
            ax.set_xlabel(selected_variable, fontsize=12)
            st.pyplot(fig)

            # 3. QQ Plot
            st.write("### QQ Plot") 
            fig = plt.figure(figsize=(10, 6))
            stats.probplot(df[selected_variable], dist="norm", plot=plt)
            plt.title(f"QQ Plot de {selected_variable}", fontsize=14)
            st.pyplot(fig)

            # Ajouter les commentaires spécifiques pour chaque variable
            if selected_variable == "age":
                st.write("""
                **Commentaires pour 'Age':**
                - La distribution de la variable 'age' s'approche d'une distribution normale malgré des distorsions aux extrémités.
                - Le jeu de données affiche une concentration des tranches d'âge 25-40 ans suivi de la tranche 40-65 ans.
                - 50% des clients ont entre 32 et 49 ans.
                - Le boxplot montre quelques valeurs extrêmes supérieures à 74 ans.
                """)
            elif selected_variable == "duration":
                st.write("""
                **Commentaires:**
                - On remarque que duration ne suit pas une distribution normale
                - 50% des appels ont une durée entre 138 et 496s (soit entre 2.3 et 8.26 min).
                - La variable présente de nombreuses valeurs extrêmes entre 1033 et 3000s.
                - Quelques valeurs très extrêmes dépassent 3000s.
                """)
            elif selected_variable == "campaign":
                st.write("""
                **Commentaires:**
                - On remarque que campaign ne suit pas une distribution normale
                - 50% du volume de contacts se situe entre 1 et 3 contacts.
                - Le boxplot montre de nombreuses valeurs extrêmes supérieures au seuil max de 6 contacts.
                - On note 3 valeurs très extrêmes supérieures à 40.
                """)
            elif selected_variable == "balance":
                st.write("""
                **Commentaires:**
                - On remarque que balance ne suit pas une distribution normale
                - 50% des clients ont une balance entre 122 et 1708€.
                - Le boxplot montre de nombreuses valeurs extrêmes concentrées entre 4087€ et 40 000€.
                - Quelques valeurs très extrêmes atteignent 81 204€.
                """)
            elif selected_variable == "pdays":
                st.write("""
                **Commentaires:**
                - On remarque que pdays ne suit pas une distribution normale
                - La valeur -1 revient constamment, signifiant que la personne n'a jamais été contactée auparavant.
                - Cette valeur a donc une signification qualitative.
                """)
            elif selected_variable == "previous":
                st.write("""
                **Commentaires:**
                - On remarque que previous ne suit pas une distribution normale
                - La valeur 0 correspond aux clients pour lesquels 'Pdays' est égal à -1.
                - Parmi les clients contactés auparavant, 50% l'ont été entre 1 et 4 fois.
                - Le boxplot montre quelques valeurs extrêmes supérieures à 8.5 contacts.
            """)

            
    


    if pages == "Analyse Multivariée" : 
        # Title and Introduction 
        st.title("Analyse Multivariée")
    
    # Define sub-pages
        sub_pages = [
            "Matrice de corrélation",
            "Tests statistiques" 
            "Évolution dans le temps"
        ]

        # Sidebar for sub-page selection
        
        if st.checkbox('**Matrice de corrélation**') :
            cor = df[['age', 'balance', 'duration', 'campaign', 'previous']].corr()
            fig, ax = plt.subplots()
            sns.heatmap(cor, annot=True, ax=ax, cmap='rainbow')
            st.write(fig)
            st.write("""Le tableau de corrélation entre toutes les variables quantitatives de notre base de donnée révèle des coefficients 
            de corrélation très proche de 0. Cela signifie que nos variables quantitatives ne sont pas corrélées entre elles.""")

        if st.checkbox("**Tests statistiques**") :
            submenu_tests = st.radio(" ", ["Tests d'ANOVA", "Tests de Chi-deux"], horizontal = True)
            
            if submenu_tests == "Tests d'ANOVA" : 
                st.header("Les variables quantitatives sont-elles liées à notre variable cible ?")
                sub_pages1 = st.radio(" ", ["Lien âge x deposit", "Lien balance x deposit", "Lien duration x deposit", "Lien campaign x deposit", "Lien previous x deposit", "Conclusion"]
                                      , key = "variable_selectbox",  horizontal=True)
                
                st.write("____________________________________")
    
                st.subheader(f"{sub_pages1}")
    
                if sub_pages1 == "Lien âge x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['age'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['age'], label='No', color='red')
                    
                    # Spécifier la taille de la police
                    plt.title('Distribution des âges selon la variable deposit', fontsize=5)  # Modifiez 10 par la taille souhaitée
                    plt.xlabel('Âge', fontsize=4)  
                    plt.ylabel('Densité', fontsize=4)  
                    plt.legend(fontsize=5) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.pyplot(fig)
                    st.write("Test Statistique d'ANOVA :")
                    
                    import statsmodels.api
                    result = statsmodels.formula.api.ols('age ~ deposit', data = df).fit()
                    table = statsmodels.api.stats.anova_lm(result)
                    st.write(table)

                    st.markdown("P_value = 0.0002  ➡️  **Il y a un lien significatif entre Age et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien balance x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['balance'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['balance'], label='No', color='red')
                    plt.title('Distribution de Balance selon la variable deposit', fontsize=5)
                    plt.xlabel('Balance', fontsize=4)  
                    plt.ylabel('Densité', fontsize=4)  
                    plt.legend(fontsize=5) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)       
    
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 9.126568e-18  ➡️  **Il y a un lien significatif entre Balance et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien duration x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['duration'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['duration'], label='No', color='red')
                    plt.title('Distribution de Duration selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  
                    plt.xlabel('Duration', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4)
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
    
                    result3 = statsmodels.formula.api.ols('duration ~ deposit', data = df).fit()
                    table3 = statsmodels.api.stats.anova_lm(result3)
                    st.write (table3)
    
                    st.markdown("P_value = 0  ➡️  **Il y a un lien significatif entre Duration et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien campaign x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['campaign'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['campaign'], label='No', color='red')
                    plt.title('Distribution de Campaign selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  
                    plt.xlabel('Campaign', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4) 
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 4.831324e-42  ➡️  **Il y a un lien significatif entre Campaign et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages1 == "Lien previous x deposit" :
                    fig = plt.figure(figsize=(4,2))
                    sns.kdeplot(df[df['deposit'] == 'yes']['previous'], label='Yes', color='blue')
                    sns.kdeplot(df[df['deposit'] == 'no']['previous'], label='No', color='red')
                    plt.title('Distribution de Previous selon la variable Deposit', fontsize=5)
                    plt.legend(fontsize=5)  # Taille de police pour la légende
                    plt.xlabel('Previous', fontsize=4) 
                    plt.ylabel('Densité', fontsize=4)                     
                    plt.yticks(fontsize=5)
                    plt.xticks(fontsize=5)
                    plt.legend()
                    st.write(fig)
    
                    st.write("Test d'ANOVA :")
                    st.markdown("P_value = 7.125338e-50  ➡️  **Il y a un lien significatif entre Previous et Deposit**") 
                    
                if sub_pages1 == "Conclusion" :
                    # st.image("/Users/admin/Desktop/BANK_APP/recap_test_anova.png")
                    st.write("Au regard des p-values (qui sont toutes inférieures à 0.05), on peut conclure que **toutes les variables quantitatives ont un lien significatif avec notre variable cible.**")
                    st.write("____________________________________")


            if submenu_tests == "Tests de Chi-deux" :     
                st.header("Les variables qualitatives sont-elles liées à notre variable cible ?")
                sub_pages2= st.radio(" ", ["Lien job x deposit", "Lien marital x deposit", "Lien education x deposit", "Lien housing x deposit", "Lien poutcome x deposit", "Conclusion"], horizontal = True)
    
                st.write("____________________________________")
    
                st.subheader(f"{sub_pages2}")
    
                if sub_pages2 == "Lien job x deposit" :
                    fig = plt.figure(figsize=(20,10))
                    sns.countplot(x="job", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
                
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['job'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Job et Deposit**") 
                    st.write("____________________________________")
    
    
                if sub_pages2 == "Lien marital x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="marital", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['marital'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Marital et Deposit**")  
                    st.write("____________________________________")
                
                
                if sub_pages2 == "Lien education x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="education", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['education'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Education et Deposit**")
                    st.write("____________________________________")
    
                
                if sub_pages2 == "Lien housing x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="housing", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test de Chi-deux :")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['housing'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Housing et Deposit**")
                    st.write("____________________________________")
    
                if sub_pages2 == "Lien poutcome x deposit" :
                    fig = plt.figure()
                    sns.countplot(x="poutcome", hue = 'deposit', data = df, palette =("g", "r"))
                    plt.legend()
                    st.pyplot(fig)
    
    
                    st.write("Test Statistique:")
    
                    from scipy.stats import chi2_contingency
                    ct = pd.crosstab(df['poutcome'], df['deposit'])
                    result = chi2_contingency(ct)
                    stat = result[0]
                    p_value = result[1]
                    st.write('Statistique: ', stat)
                    st.write('P_value: ', p_value)
    
                    st.write("**Il y a une dépendance entre Poutcome et Deposit**")  
                    st.write("____________________________________")
    
                if sub_pages2 == "Conclusion" :
    
                    # st.image("/Users/admin/Desktop/BANK_APP/recap_Chi-deux.png")
                    st.write("Au regard des p-values (qui sont toutes inférieures à 0.05), on peut conclure que **toutes les variables qualitatives ont un lien significatif avec notre variable cible.**")
                    st.write("____________________________________")


        if st.checkbox("**Évolution dans le temps**"):  
            st.header("Analyse de l'évolution de la variable deposit dans le temps")
            sub_pages3= st.radio(" ", ["Deposit x month", "Deposit x year", "Deposit x weekday"], horizontal = True)

            st.write("____________________________________")

            st.subheader(f"Analyse du {sub_pages3}")
            
            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in df["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            df["year"] = liste_annee

            df['date'] = df['day'].astype(str)+ '-'+ df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df['date']= pd.to_datetime(df['date'])

            df["weekday"] = df["date"].dt.weekday
            dic = {0 : "Lundi",
            1 : "Mardi",
            2 : "Mercredi",
            3 : "Jeudi",
            4 : "Vendredi",
            5 : "Samedi",
            6 : "Dimanche"}
            df["weekday"] = df["weekday"].replace(dic)


            df['month_year'] = df['month'].astype(str)+ '-'+ df['year'].astype(str)
            df_order_month = df.copy()
            df_order_month = df_order_month.sort_values(by='date')
            df_order_month["month_year"] = df_order_month["month"].astype(str)+ '-'+ df_order_month["year"].astype(str)

            #creation de la colonne Client_Category_M selon pdays
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            df['Client_Category_M'] = pd.cut(df['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            df['Client_Category_M'] = df['Client_Category_M'].astype('object')

            if sub_pages3 == "Deposit x month":
                fig = plt.figure(figsize=(20,10))
                sns.countplot(x='month_year', hue='deposit', data=df_order_month, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon les mois")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous pouvons remarquer qu'au début de notre période d'étude la proportion des clients qui
                ont souscrit à un dépôt à terme est inférieur à celle qui n'y ont pas souscrit.""")

            if sub_pages3 == "Deposit x year": 
                fig = plt.figure()
                sns.countplot(x='year', hue='deposit', data=df, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon l'année")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous pouvons remarquer ici que la proportion des clients (ayant souscrit ou non à un dépôt à terme)
                est supérieur durant l'année 2013 que 2014. Ceci serait surement dù à la période de l'étude (7 mois en 2013 et 5 mois en 2014) """)


            if sub_pages3 == "Deposit x weekday":
                fig = plt.figure()
                sns.countplot(x="weekday", hue = 'deposit', data = df, palette =("g", "r"))
                plt.title("Évolution de notre variable cible selon les jours de la semaine")
                plt.legend()
                st.pyplot(fig)
                st.write("""Nous remarquons qu'en général les clients souscrivent au dépôt à terme le week-end .""")
        

    if pages == "Profiling" :  
        if st.checkbox("Analyses"):
    
            # Title and Introduction
            st.title("Profil des clients 'Deposit YES'")
            
            # Filter the dataset
            dff = df[df['job'] != "unknown"]  # Remove rows with unknown job
            dff = dff[dff['education'] != "unknown"]  # Remove rows with unknown education
    
            # Replace 'unknown' in poutcome with NaN, then fill with the mode
            dff['poutcome2'] = dff['poutcome'].replace('unknown', np.nan)
            dff['poutcome2'] = dff['poutcome2'].fillna(dff['poutcome2'].mode()[0])
    
            # Drop the 'contact' column as it's not needed
            dff = dff.drop(['contact'], axis=1)
    
            #  Creation de categorie de client
    
            liste =[]
    
            for i in dff["pdays"] :
                if i == -1 :
                    liste.append("new_prospect")
                elif i != -1 :
                    liste.append("old_prospect")
    
            dff["type_prospect"] = liste
    
    
            # Filter clients who have subscribed
            clients_yes = dff[dff["deposit"] == "yes"]
            
    
            # Display the number of subscribed clients
            st.text(f"Nombre de clients ayant souscrit à un compte de dépôt à terme : {clients_yes.shape[0]}")
    
            # Define sub-pages
            sub_pages = st.sidebar.radio(" ", [
                "Age et Job",
                "Statut Matrimonial et Education",
                "Bancaire",
                "Campagnes Marketing",
                "Temporel",
                "Duration"
            ], horizontal = True)
            
            # Sidebar for sub-page selection
    
            # Logic for each sub-page
            if sub_pages == "Age et Job":
                st.write("### Analyse: Age et Job")
                plt.figure(figsize=(10, 6), dpi=120)
                sns.histplot(clients_yes['age'], kde=False, bins=30)
                plt.title("Distribution de l'âge des clients")
                plt.xlabel("Âge des clients")
                plt.ylabel("Nombre de clients")
            
            # Display the plot in Streamlit
                st.pyplot(plt)
    
       # Calcul du nombre de clients par job
                total_client_job = clients_yes.groupby('job').size().reset_index(name='Total Clients')
    
                # Calcul de la moyenne, du minimum, du maximum de la variable 'age' par job
                group_age_job = clients_yes.groupby('job')['age'].agg(['mean', 'min', 'max']).reset_index()
    
                # Renommage des colonnes
                group_age_job.columns = ['job', 'Age Moyen', 'Age Minimum', 'Age Maximum']
    
                # Fusion des deux DataFrames sur la colonne 'job'
                summary = pd.merge(total_client_job, group_age_job, on='job')
    
                # Triage par ordre décroissant du nombre de clients
                summary = summary.sort_values(by='Total Clients', ascending=False)
    
                # Réinitialiser l'index et supprimer la colonne d'index
                summary = summary.reset_index(drop=True)
    
                 # Affichage du DataFrame final dans Streamlit sans la colonne d'index
                st.write("### Résumé des clients par job avec les statistiques d'âge:")
                st.dataframe(summary)
                st.text("Nous remarquons sur ce tableau qu'il y a une grande diversification des âges pour tous les groupes.")
                st.write("____________________________________")
                
            elif sub_pages == "Statut Matrimonial et Education":
                st.write("### Analyse: Statut Matrimonial et Education")
        # --- Statut matrimonial ---
                marital_counts = clients_yes['marital'].value_counts()
                marital_percentage = marital_counts / marital_counts.sum() * 100
                plt.figure(figsize=(10, 8))
                sns.barplot(x=marital_percentage.index, y=marital_percentage.values, color='skyblue')
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(marital_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et étiquettes des axes
                plt.title("Distribution du statut matrimonial des clients qui ont souscrit à un dépôt à terme")
                plt.xlabel("Statut matrimonial")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique avec Streamlit
                st.pyplot(plt)
    
                # --- Niveau d'éducation ---
                education_counts = clients_yes['education'].value_counts()
                education_percentage = education_counts / education_counts.sum() * 100
                plt.figure(figsize=(10, 8))
                sns.barplot(x=education_percentage.index, y=education_percentage.values, color='skyblue')
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(education_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et étiquettes des axes
                plt.title("Distribution du niveau académique des clients qui ont souscrit à un dépôt à terme")
                plt.xlabel("Education")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique avec Streamlit
                st.pyplot(plt)
    
                # Texte explicatif
                st.text("Nous observons que la majorité des clients sont mariés, suivis par un groupe de clients célibataires. Les niveaux d'éducation des clients sont le secondaire et le tertiaire. Ceci montre que les clients détenant le DAT (dépôt à terme) ont un certain niveau académique.")
                st.write("____________________________________")

            elif sub_pages == "Bancaire":
                st.header("Analyse: Bancaire")
                st.subheader("Balance du compte")
            
                # Séparation des clients en fonction du solde
                clients_positif = clients_yes[clients_yes['balance'] > 0]
                clients_negatif = clients_yes[clients_yes['balance'] <= 0]
                
                nb_clients_positif = len(clients_positif)
                nb_clients_negatif = len(clients_negatif)
    
                pourcentage_positif = (nb_clients_positif / len(clients_yes)) * 100
                pourcentage_negatif = (nb_clients_negatif / len(clients_yes)) * 100
    
                
    
                # Labels pour les groupes
                labels = ['Solde positif', 'Solde négatif ou nul']
                counts = [nb_clients_positif, nb_clients_negatif]
    
                # Créer un DataFrame temporaire pour le plot
                data = pd.DataFrame({'Type de solde': labels, 'Nombre de clients': counts})
                
    
                # Créer un bar plot pour comparer les deux groupes
                plt.figure(figsize=(9, 6), dpi=100)
                sns.barplot(x='Type de solde', y='Nombre de clients', data=data, palette="pastel")
                plt.title("Comparaison des clients avec un solde positif et un solde négatif ou nul")
                plt.xlabel("Type de solde")
                plt.ylabel("Nombre de clients")
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(counts):
                    plt.text(i, v + 5, f"{(v / len(clients_yes)) * 100:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                st.pyplot(plt)
                st.write(f"Pourcentage de clients avec un solde positif : {pourcentage_positif:.2f}%")
                st.write(f"Pourcentage de clients avec un solde négatif ou nul : {pourcentage_negatif:.2f}%")
    
                st.subheader("Loan/Housing/default")
    
                # Statistiques pour 'housing'
                housing_counts = clients_yes['housing'].value_counts()
                housing_percentage = housing_counts / housing_counts.sum() * 100
                housing_stats = pd.DataFrame({
                    'Housing Status': housing_counts.index,
                    'Count': housing_counts.values,
                    'Percentage': housing_percentage.values
                })
    
                # Statistiques pour 'loan'
                loan_counts = clients_yes['loan'].value_counts()
                loan_percentage = loan_counts / loan_counts.sum() * 100
                loan_stats = pd.DataFrame({
                    'Loan Status': loan_counts.index,
                    'Count': loan_counts.values,
                    'Percentage': loan_percentage.values
                })
    
                # Statistiques pour 'default'
                default_counts = clients_yes['default'].value_counts()
                default_percentage = default_counts / default_counts.sum() * 100
                default_stats = pd.DataFrame({
                    'Default Status': default_counts.index,
                    'Count': default_counts.values,
                    'Percentage': default_percentage.values
                })
    
                
    
                # --- Bar plot pour housing ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=housing_percentage.index, y=housing_percentage.values, palette="pastel")
                plt.title("Distribution des prêts immobiliers parmi les clients ayant un dépôt à terme")
                plt.xlabel("Housing")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(housing_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                # --- Bar plot pour loan ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=loan_percentage.index, y=loan_percentage.values, palette="pastel")
                plt.title("Distribution des prêts personnels parmi les clients ayant un dépôt à terme")
                plt.xlabel("Loan")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(loan_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                # --- Bar plot pour default ---
                plt.figure(figsize=(9, 6))
                sns.barplot(x=default_percentage.index, y=default_percentage.values, palette="pastel")
                plt.title("Distribution de défaut de paiement parmi les clients ayant un dépôt à terme")
                plt.xlabel("Default")
                plt.ylabel("Pourcentage de clients (%)")
                
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(default_percentage.values):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
                
                st.pyplot(plt)
    
                st.text("Parmi les clients qui ont un DAT :")
                st.text("Plus de 60% des clients n'ont pas de prêt immobilier.")
                st.text("90% des clients n'ont pas de prêt personnel.")
                st.text("99% des clients ayant des engagements bancaires ne sont pas en défaut de paiement.")
                st.write("____________________________________")

    
        
            elif sub_pages == "Campagnes Marketing":
                st.write("### Analyse: Caractéristiques des Campagnes marketing")
    
                if st.checkbox("Type de clients"):
                    st.write("Clients Jamais contactés ou déjà contactés lors de la précédente campagne marketing") 
                    # Nombre de clients par type de prospect
                    prospect_counts = clients_yes["type_prospect"].value_counts()
    
                    # Affichage des résultats
                    st.dataframe(prospect_counts)
    
                    # Fonction pour tracer les barres
            
            
            
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        fig, ax = plt.subplots(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue', ax=ax)
                        
                        # Ajouter les annotations de pourcentage sur les barres
                        for p in ax.patches:  # Pour chaque barre du plot
                            ax.annotate(f'{p.get_height():.1f}%', 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='bottom') 
                
                        plt.title(f"Distribution de {column} (%)", fontsize=15) 
                        plt.xlabel(xlabel, fontsize=12)
                        plt.ylabel("Percentage (%)", fontsize=12)
                        
                        st.pyplot(fig)
                        plt.clf()  
                
                    plot_percentage(clients_yes, "type_prospect", "Type de prospect")

                    st.write("On voit ici que plus de 60% des clients qui ont souscrit au DAT sont de nouveaux prospects.")
                    st.write("____________________________________")

                
                if st.checkbox("Poutcome"):
                    st.write("Résultat de la précédente campagne marketing")  
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        fig, ax = plt.subplots(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue', ax=ax)
                     
                        # Ajouter les annotations de pourcentage sur les barres
                        for p in ax.patches:  # Pour chaque barre
                            ax.annotate(f'{p.get_height():.1f}%', 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='bottom') 
                    
                        # Paramètres du graphique
                        plt.title(f"Distribution de {column} (%)", fontsize=15) 
                        plt.xlabel(xlabel, fontsize=12)
                        plt.ylabel("Percentage (%)", fontsize=12)
                        
                        # Afficher le graphique dans Streamlit
                        st.pyplot(fig)
                        plt.clf()  
                    
                    # Appel de la fonction pour tracer le graphique
                    plot_percentage(clients_yes, "poutcome2", "Poutcome: Résultat de la précédente campagne")

                    st.write("Plus de 70 % des clients précédemment contactés, qui avaient refusé l'offre lors de la campagne précédente, ont accepté de souscrire à cette nouvelle campagne de dépôt à terme.")
                    st.write("____________________________________")

                if st.checkbox("Previous"):
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        plt.figure(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)")
                        plt.xlabel(xlabel)
                        plt.ylabel("Percentage (%)")
                        plt.xticks(rotation=45)  
                        st.pyplot(plt)
                        plt.clf()  
                    st.write("Nombre de contacts réalisés avec le client avant la campagne")   
                    plot_percentage(clients_yes, "previous", "Nombre de contact réalisé avant la campagne")

                    st.write("Plus de 60% des clients qui ont souscrit au DAT n'avaient jamais été contacté par la banque avant cette campagne.")
    
                if st.checkbox("Campaign"):
                    #fonction
                    def plot_percentage(data, column, xlabel):
                        if column not in data.columns:
                            st.error(f"The column '{column}' does not exist in the dataset.")
                            return
            
                    # Calculate percentages
                        counts = data[column].value_counts(normalize=True) * 100
            
                        # Barplot de distribution
                        plt.figure(figsize=(9, 6))
                        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
                        plt.title(f"Distribution de {column} (%)")
                        plt.xlabel(xlabel)
                        plt.ylabel("Percentage (%)")
                        plt.xticks(rotation=45)  
                        st.pyplot(plt)
                        plt.clf()  
                    st.write("Nombre de contacts réalisés avec le client pendant la campagne") 
                    plot_percentage(clients_yes, "campaign", "Nombre de contact réalisé pendant la campagne")
                    st.write("La plus grande proportion des clients qui ont souscrit au DAT a été contactée une fois pendant cette campagne. Donc en un appel le client a accepté l'offre.")
                    st.write("____________________________________")

            elif sub_pages == "Temporel":
                st.write("### Analyse: Temporel")
    
                liste_annee =[]
                for i in clients_yes["month"] :
                    if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                        liste_annee.append("2013")
                    elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                        liste_annee.append("2014")
                clients_yes["year"] = liste_annee
                clients_yes['date'] = clients_yes['day'].astype(str)+ '-'+ clients_yes['month'].astype(str)+ '-'+ clients_yes['year'].astype(str)
                clients_yes['date']= pd.to_datetime(clients_yes['date'])
                clients_yes["weekday"] = clients_yes["date"].dt.weekday
                dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
                clients_yes["weekday"] = clients_yes["weekday"].replace(dic)
               
    
                # Mois
                month_year_counts = clients_yes['month'].value_counts()
                month_year_percentage = month_year_counts / month_year_counts.sum() * 100
                plt.figure(figsize=(8, 5))
                sns.barplot(x=month_year_percentage.index, y=month_year_percentage.values, color='skyblue')
                plt.title("Distribution des mois où les clients  ont souscrit à un dépôt à terme")
                plt.xlabel("Mois")
                plt.ylabel("Pourcentage de clients (%)")
                plt.xticks(rotation=90)
                st.pyplot(plt)
    
                #  Jour de la semaine
                weekday_counts = clients_yes['weekday'].value_counts()
                weekday_percentage = weekday_counts / weekday_counts.sum() * 100
                plt.figure(figsize=(8, 5))
                sns.barplot(x=weekday_percentage.index, y=weekday_percentage.values, color='skyblue')
                plt.title("Distribution des jours de la semaine où les clients ont souscrit à un dépôt à terme")
                plt.xlabel("weekday")
                plt.ylabel("Pourcentage de clients (%)")
                st.pyplot(plt)
    
                st.text("Les périodes où les clients sont susceptibles de souscrire sont le printemps et l'été. Et les jours sont par ordre de souscription : dimanche, mardi, mercredi, lundi, jeudi, vendredi et samedi.")
                st.write("____________________________________")

    
            elif sub_pages == "Duration":
                st.write("### Analyse: Duration")
            
                # Conversion de la durée en minutes
                clients_yes = clients_yes.copy()
                clients_yes['duration_minutes'] = clients_yes['duration'] / 60
    
                # Calcul des valeurs de référence
                mean_duration = clients_yes['duration_minutes'].mean()
                min_duration = clients_yes['duration_minutes'].min()
                max_duration = clients_yes['duration_minutes'].max()
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure au minimum
                nb_clients_min_or_more = len(clients_yes[clients_yes['duration_minutes'] >= min_duration])
                pourcentage_min_or_more = (nb_clients_min_or_more / len(clients_yes)) * 100
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure au maximum
                nb_clients_max_or_more = len(clients_yes[clients_yes['duration_minutes'] >= max_duration])
                pourcentage_max_or_more = (nb_clients_max_or_more / len(clients_yes)) * 100
    
                # Calcul du pourcentage de clients avec une durée égale ou supérieure à la moyenne
                nb_clients_mean_or_more = len(clients_yes[clients_yes['duration_minutes'] >= mean_duration])
                pourcentage_mean_or_more = (nb_clients_mean_or_more / len(clients_yes)) * 100
    
                # Affichage des résultats sous forme textuelle
                st.write(f"Durée moyenne (minutes) : {mean_duration:.2f}")
                st.write(f"Durée minimum (minutes) : {min_duration:.2f}")
                st.write(f"Durée maximum (minutes) : {max_duration:.2f}")
    
    
    
                # Création d'un DataFrame pour les pourcentages à afficher dans le graphique
                duration_stats = {
                    'Durée': ['Moyenne', 'Minimum', 'Maximum'],
                    'Pourcentage': [pourcentage_mean_or_more, pourcentage_min_or_more, pourcentage_max_or_more]
                }
                duration_df = pd.DataFrame(duration_stats)
    
                # Plot des pourcentages
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Durée', y='Pourcentage', data=duration_df, palette="pastel")
    
                # Ajouter les pourcentages sur les barres
                for i, v in enumerate(duration_df['Pourcentage']):
                    plt.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10, color='black')
    
                # Titre et labels
                plt.title("Pourcentage de clients en fonction de la durée d'appel")
                plt.xlabel("Critère de durée")
                plt.ylabel("Pourcentage de clients (%)")
    
                # Affichage du graphique
                st.pyplot(plt)
    
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale à la moyenne : {pourcentage_mean_or_more:.2f}%")
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale au minimum : {pourcentage_min_or_more:.2f}%")
                st.write(f"Pourcentage de clients avec une durée supérieure ou égale au maximum : {pourcentage_max_or_more:.2f}%")
                st.write("____________________________________")



        if st.checkbox("Récapitulatif"):
            st.write("#### Le profil des clients ayant souscrit au produit DAT de la banque est le suivant :")
            st.write("* Clients **âgés entre 25 et 60 ans** avec des métiers de **manager, technicien, ouvrier, ou travaillant dans l'administration.**")
            st.write("* Ils sont **mariés** pour la plupart et ont un niveau **académique secondaire ou tertiaire.**")
            st.write("* La majorité des clients n'ont **pas d'engagement bancaire** (prêt personnel, prêt immobilier) et ne sont **pas en défaut de paiement.**")
            st.write("* Ils n'ont, pour la plupart, **jamais été contacté par la banque.**")
            st.write("* Ils souscrivent au DAT dans les périodes **fin printemps / l'été**, principalement, dans l'ordre, **le dimanche, mardi, mercredi, lundi.**")
            st.write("* Et la durée moyenne des appels pour convaincre un client de souscrire à un DAT est de **9 minutes.**")
        
if selected == "Pre-processing":  
    st.title("PRÉ-PROCESSING")
    option_submenu3 = st.radio(" ", ["**AVANT SÉPARATION DES DONNÉES**", "**APRÈS SÉPARATION DES DONNÉES**"], horizontal = True)
        
        
    if option_submenu3 == '**AVANT SÉPARATION DES DONNÉES**':
        submenupages=st.radio(" ", ["Suppression de lignes", "Création de colonnes", "Suppression de colonnes", "Gestion des Unknowns"], horizontal = True)

        dffpre_pros = df.copy()
        dffpre_pros2 = df.copy()
   
        if submenupages == "Suppression de lignes" :            
            st.subheader("Filtre sur la colonne 'age'")
            st.markdown("Notre analyse univariée a montré des **valeurs extrêmes au dessus de 74 ans.** \n\
            Nous avons décidé de retirer ces lignes de notre dataset.")
            
            dffpre_pros = dffpre_pros[dffpre_pros['age'] < 75]
            count_age_sup = df[df['age'] > 74.5].shape[0]
            st.write("Résultat =", count_age_sup,"**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'balance'")
            st.markdown("Pour balance, nous avons également constaté des **valeurs extrêmes** pour **les valeurs inférieures à -2257** et les **valeurs supérieures à 4087**. \n\
            Nous retirons ces lignes.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] > -2257]
            dffpre_pros = dffpre_pros.loc[dffpre_pros["balance"] < 4087]
            count_balance_sup = df[df['balance'] < -2257].shape[0]
            count_balance_inf = df[df['balance'] > 4087].shape[0]
            total_balance_count = count_balance_sup + count_balance_inf
            st.write("Résultat =", total_balance_count, "**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'campaign'")
            st.markdown("La variable campaign a également montré des **valeurs extrêmes pour les valeurs supérieures à 6**.  \n\
            Nous retirons également ces lignes.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["campaign"] < 6]
            count_campaign_sup = df[df['campaign'] > 6].shape[0]
            st.write("Résultat", count_campaign_sup,"**lignes supprimées**")
            
            st.subheader("Filtre sur la colonne 'previous'")
            st.markdown("Nous avons également constaté des **valeurs extrêmes pour les valeurs supérieures à 2**. \n\
            Nous retirons également ces lignes de notre dataset.")
            dffpre_pros = dffpre_pros.loc[dffpre_pros["previous"] < 2.5]
            count_previous_sup = df[df['previous'] > 2.5].shape[0]
            st.write("Résultat", count_previous_sup,"**lignes supprimées**")
            
            st.write("____________________________________")

            st.subheader("Résultat:")
            count_sup_lignes = df.shape[0] - dffpre_pros.shape[0]
            st.write("Nombre total de lignes supprimées = ", count_sup_lignes)
            nb_lignes = dffpre_pros.shape[0]
            st.write("**Notre jeu de données filtré compte désormais ", nb_lignes, "lignes.**")

        if submenupages == "Création de colonnes" :   
            st.subheader("Création de la colonne 'Client_Category'")
            st.write("La colonne **'pdays'** indique le nombre de jours depuis le dernier contact avec un client lors de la campagne précédente, mais contient souvent **la valeur -1, signalant des clients jamais contactés**.")
            st.write("Pour distinguer les clients ayant été contactés de ceux qui ne l'ont pas été, une nouvelle colonne **'Client_Category_M'** est créée à partir de 'pdays'.")
            st.markdown("Cette nouvelle colonne nouvellement créée comprend 3 valeurs :  \n\
            1. **Prospect** = clients qui n'ont jamais été contacté lors de la précédente campagne  \n\
            2. **Reached-6M** = clients contactés il y a moins de 6 mois lors de la précédente campagne  \n\
            3. **Reached+6M** = clients contactés il y a plus de 6 mois lors de la précédente campagne")

            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros['Client_Category_M'] = pd.cut(dffpre_pros['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros['Client_Category_M'] = dffpre_pros['Client_Category_M'].astype('object')
                        
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros.head(10))
            
            st.subheader("Création de la colonne 'weekday'")
            st.markdown("Avant de pouvoir créer la colonne weekday, nous devons passer par deux étapes :  \n\
            1. **ajouter une colonne year** : les données fournies par la banque portugaises sont datées de juin 2014. Nous en déduisons que les mois allant de juin à décembre correspondent à l'année 2013 et que les mois allant de janvier à mai correspondent à l'année 2014  \n\
            2. **ajouter une colonne date au format datetime** : cela est désormais possibles grâce aux colonnes mois, day et year")
            
            st.markdown("**Nous pouvons alors créer la colonne weekday grâce à la fonction 'dt.weekday'**")
            
            #creation des colonnes year, month_year, date, weekday
            liste_annee =[]
            for i in dffpre_pros["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros["year"] = liste_annee
    
            dffpre_pros['date'] = dffpre_pros['day'].astype(str)+ '-'+ dffpre_pros['month'].astype(str)+ '-'+ dffpre_pros['year'].astype(str)
            dffpre_pros['date']= pd.to_datetime(dffpre_pros['date'])
    
            dffpre_pros["weekday"] = dffpre_pros["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros["weekday"] = dffpre_pros["weekday"].replace(dic)
            
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros.head(10))
            
        
        if submenupages == "Suppression de colonnes" :
            st.subheader("Suppressions de colonnes")
        
            st.write("- La colonne contact ne contribuerait pas de manière significative à la compréhension des données, nous décidons donc de la supprimer.")             
            st.write("- Puisque nous avons créé la colonne Client_Category à partir de la colonne 'pdays', nous supprimons la colonne 'pdays'") 
            st.write("- Puisque nous avons créé la colonne weeday à partir de la colonne 'date', nous supprimons la colonne 'day' ainsi que la colonne date qui nous a uniquement servi à crééer notre colonne weekday.")     
            st.write("- Enfin, nous nous pouvons supprimer la colonne 'year' car les années 2013 et 2014 ne sont pas complètes, nous ne pouvons donc pas les comparer.")

                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            

            st.write("____________________________________")

            st.subheader("Résultat:")
            colonnes_count = dffpre_pros2.shape[1]
            nb_lignes = dffpre_pros2.shape[0]
            st.write("Notre dataset compte désormais :", colonnes_count, "colonnes et", nb_lignes, "lignes.")
            
            # Affichage du nouveau dataset
            st.dataframe(dffpre_pros2.head(5))


        if submenupages == "Gestion des Unknowns" : 
            st.subheader("Les colonnes 'job', 'education' et 'poutcome' contiennent des valeurs 'unknown', il nous faut donc les remplacer.")
            st.write("Pour cela nous allons tout d'abord transformer les valeurs 'unknown' en 'nan'.")
            
            # Transformation des 'unknown' en NaN déjà fait plus haut
                                    
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
            st.dataframe(dffpre_pros2.isna().sum())
            
            st.markdown("Nous nous occuperons du remplacement de ces NAns par la suite, une fois le jeu de donnée séparé en jeu d'entraînement et de test.  \n\
            **Cela dans le but de s'assurer que la même transformation des Nans est appliquée au jeu de données Train et Test.**")
            

    if option_submenu3 == '**APRÈS SÉPARATION DES DONNÉES**':
        submenupages2 = st.radio(" ", ["Séparation train test", "Traitement des valeurs manquantes", "Standardisation des variables", "Encodage"], horizontal = True)
         
        if submenupages2 == "Séparation train test" :
            st.subheader("Séparation train test")
            st.write("Nous appliquons un **ratio de 80/20 pour notre train test split** : 80% des données en Train et 20% en Test.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)


            colonnes_count = X_train_pre_pros2.shape[1]
            nb_lignes = X_train_pre_pros2.shape[0]
            st.write("Le dataframe X_train compte :", colonnes_count, "colonnes et", nb_lignes, "lignes :")
            st.dataframe(X_train_pre_pros2.head())
                
            colonnes_count = X_test_pre_pros2.shape[1]
            nb_lignes = X_test_pre_pros2.shape[0]
            st.write("Le dataframe X_test compte :", colonnes_count, "colonnes et", nb_lignes, "lignes :")
            st.dataframe(X_test_pre_pros2.head())
                
        if submenupages2 == "Traitement des valeurs manquantes" :    
            st.subheader("Traitement des valeurs manquantes")
            st.write("Pour la **colonne job**, on remplace les Nans par le **mode** de la variable.")
            st.write("S'agissant des **colonnes 'education' et 'poutcome'**, puisque le nombre de Nans est plus élevé, nous avons décidé de les remplacer en utilisant la **méthode de remplissage par propagation** : chaque Nan est remplacé par la valeur de la ligne suivante (pour la dernière ligne on utilise le mode de la variable).") 
            st.write("Ce processus est appliqué à X_train et X_test.")

            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])

            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])

            col1, col2 = st.columns(2)
            with col1 :
             st.write("Vérification sur X_train, reste-t-il des Nans ?")
             st.dataframe(X_train_pre_pros2.isna().sum())
            with col2 :   
             st.write("Vérification sur X_test, reste-t-il des Nans ?")
             st.dataframe(X_test_pre_pros2.isna().sum())

                
        if submenupages2 == "Standardisation des variables" :    
            st.subheader("Standardisation des variables")
            st.write("On **standardise les variables quantitatives** à l'aide de la **fonction StandardScaler**.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])
                
            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])

            col1, col2 = st.columns(2)
            with col1 :
             st.write("Vérification sur X_train, reste-t-il des Nans ?")
             st.dataframe(X_train_pre_pros2.isna().sum())
            with col2 :   
             st.write("Vérification sur X_test, reste-t-il des Nans ?")
             st.dataframe(X_test_pre_pros2.isna().sum())

                
        if submenupages2 == "Encodage" :    
            st.subheader("Encodage")
            st.write("On encode la **variable cible** avec le **Label Encoder**.")
            dffpre_pros2 = df.copy()                        
            dffpre_pros2 = dffpre_pros2[dffpre_pros2['age'] < 75]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] > -2257]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["balance"] < 4087]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["campaign"] < 6]
            dffpre_pros2 = dffpre_pros2.loc[dffpre_pros2["previous"] < 2.5]
            
            bins = [-2, -1, 180, 855]
            labels = ['Prospect', 'Reached-6M', 'Reached+6M']
            dffpre_pros2['Client_Category_M'] = pd.cut(dffpre_pros2['pdays'], bins=bins, labels=labels)
            # Transformation de 'Client_Category' en type 'objet'
            dffpre_pros2['Client_Category_M'] = dffpre_pros2['Client_Category_M'].astype('object')
            
            liste_annee =[]
            for i in dffpre_pros2["month"] :
                if i == "jun" or i == "jul" or i == "aug" or i == "sep" or i == "oct" or i == "nov" or i == "dec" :
                    liste_annee.append("2013")
                elif i == "jan" or i == "feb" or i == "mar" or i =="apr" or i =="may" :
                    liste_annee.append("2014")
            dffpre_pros2["year"] = liste_annee
    
            dffpre_pros2['date'] = dffpre_pros2['day'].astype(str)+ '-'+ dffpre_pros2['month'].astype(str)+ '-'+ dffpre_pros2['year'].astype(str)
            dffpre_pros2['date']= pd.to_datetime(dffpre_pros2['date'])
    
            dffpre_pros2["weekday"] = dffpre_pros2["date"].dt.weekday
            dic = {0 : "Lundi", 1 : "Mardi", 2 : "Mercredi", 3 : "Jeudi", 4 : "Vendredi", 5 : "Samedi", 6 : "Dimanche"}
            dffpre_pros2["weekday"] = dffpre_pros2["weekday"].replace(dic)
            dffpre_pros2 = dffpre_pros2.drop(['contact'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['pdays'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['day'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['date'], axis=1)
            dffpre_pros2 = dffpre_pros2.drop(['year'], axis=1)
            # Transformation des 'unknown' en NaN
            dffpre_pros2['job'] = dffpre_pros2['job'].replace('unknown', np.nan)
            dffpre_pros2['education'] = dffpre_pros2['education'].replace('unknown', np.nan)
            dffpre_pros2['poutcome'] = dffpre_pros2['poutcome'].replace('unknown', np.nan)
            
                
            # Séparation des données en un jeu de variables explicatives X et variable cible y
            X_pre_pros2 = dffpre_pros2.drop('deposit', axis = 1)
            y_pre_pros2 = dffpre_pros2['deposit']

            # Séparation des données en un jeu d'entrainement et jeu de test
            X_train_pre_pros2, X_test_pre_pros2, y_train_pre_pros2, y_test_pre_pros2 = train_test_split(X_pre_pros2, y_pre_pros2, test_size = 0.20, random_state = 48)

            # Remplacement des NaNs par le mode:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_train_pre_pros2.loc[:,['job']] = imputer.fit_transform(X_train_pre_pros2[['job']])
            X_test_pre_pros2.loc[:,['job']] = imputer.transform(X_test_pre_pros2[['job']])
                
            # On remplace les NaaN de 'poutcome' & 'education' avec la méthode de remplissage par propagation et on l'applique au X_train et X_test :
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(method ='bfill')
            X_train_pre_pros2['poutcome'] = X_train_pre_pros2['poutcome'].fillna(X_train_pre_pros2['poutcome'].mode()[0])

            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(method ='bfill')
            X_test_pre_pros2['poutcome'] = X_test_pre_pros2['poutcome'].fillna(X_test_pre_pros2['poutcome'].mode()[0])

            # On fait de même pour les NaaN de 'education'
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(method ='bfill')
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].fillna(X_train_pre_pros2['education'].mode()[0])

            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(method ='bfill')
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].fillna(X_test_pre_pros2['education'].mode()[0])

            # Standardisation des variables quantitatives:
            scaler = StandardScaler()
            cols_num = ['age', 'balance', 'duration', 'campaign', 'previous']
            X_train_pre_pros2 [cols_num] = scaler.fit_transform(X_train_pre_pros2 [cols_num])
            X_test_pre_pros2 [cols_num] = scaler.transform (X_test_pre_pros2 [cols_num])

            # Encodage de la variable Cible 'deposit':
            le = LabelEncoder()
            y_train_pre_pros2 = le.fit_transform(y_train_pre_pros2)
            y_test_pre_pros2 = le.transform(y_test_pre_pros2)
                
            st.write("S'agissant des variables qualitatives à 2 modalités **'default'**, **'housing'** et **'loan'**, on encode avec le **One Hot Encoder**.")
            # Encodage des variables explicatives de type 'objet'
            oneh = OneHotEncoder(drop = 'first', sparse_output = False)
            cat1 = ['default', 'housing','loan']
            X_train_pre_pros2.loc[:, cat1] = oneh.fit_transform(X_train_pre_pros2[cat1])
            X_test_pre_pros2.loc[:, cat1] = oneh.transform(X_test_pre_pros2[cat1])

            X_train_pre_pros2[cat1] = X_train_pre_pros2[cat1].astype('int64')
            X_test_pre_pros2[cat1] = X_test_pre_pros2[cat1].astype('int64')
                
            st.write("Pour les variables ordinales **'education'** et **'Client_Category'**, on **remplace les modalités par des nombres** en tenant compte de l'ordre.")
                
            # 'education' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['education'] = X_train_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])
            X_test_pre_pros2['education'] = X_test_pre_pros2['education'].replace(['primary', 'secondary', 'tertiary'], [0, 1, 2])

            # 'Client_Category_M' est une variable catégorielle ordinale, remplacer les modalités de la variable par des nombres, en gardant l'ordre initial
            X_train_pre_pros2['Client_Category_M'] = X_train_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])
            X_test_pre_pros2['Client_Category_M'] = X_test_pre_pros2['Client_Category_M'].replace(['Prospect', 'Reached-6M', 'Reached+6M'], [0, 1, 2])


            st.write("Pour les **variables catégorielles à plus de 2 modalités** on applique le **get dummies** sur X_train et X_test.")
                
            # Encoder les variables à plus de 2 modalités 'job', 'marital', 'poutome', 'month', 'weekday' pour X_train
            dummies = pd.get_dummies(X_train_pre_pros2['job'], prefix='job').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('job', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['job'], prefix='job').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('job', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['marital'], prefix='marital').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('marital', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['marital'], prefix='marital').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('marital', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['poutcome'], prefix='poutcome').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('poutcome', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['poutcome'], prefix='poutcome').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('poutcome', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['month'], prefix='month').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('month', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['month'], prefix='month').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('month', axis=1), dummies], axis=1)

            dummies = pd.get_dummies(X_train_pre_pros2['weekday'], prefix='weekday').astype(int)
            X_train_pre_pros2 = pd.concat([X_train_pre_pros2.drop('weekday', axis=1), dummies], axis=1)
            dummies = pd.get_dummies(X_test_pre_pros2['weekday'], prefix='weekday').astype(int)
            X_test_pre_pros2 = pd.concat([X_test_pre_pros2.drop('weekday', axis=1), dummies], axis=1)


            st.write("Dataframe final X_train : ")
            st.dataframe(X_train_pre_pros2.head())
                
            #Afficher les dimensions des jeux reconstitués.
            st.write("**Dimensions du jeu d'entraînement :**",X_train_pre_pros2.shape)
            st.write("")
                
            st.write("Dataframe final X_test : ")
            st.dataframe(X_test_pre_pros2.head())
            st.write("**Dimensions du jeu de test :**",X_test_pre_pros2.shape)
                

if selected == "Modélisation":
    st.title("MODÉLISATION")
    st.sidebar.title("SOUS MENU MODÉLISATION")  
    pages=["Introduction", "Modélisation avec Duration", "Modélisation sans Duration"]
    page=st.sidebar.radio('Afficher', pages)
 
    
    #RÉSULTAT DES MODÈLES SANS PARAMÈTRES
    # ON A PRÉCÉDEMMENT FAIT TOURNER UN CODE POUR ENREGISTRER LES MODÈLES SANS PARAMÈTRES DANS JOBLIB
    

    #Liste des modèles enregistrés et leurs noms
    model_files = {
        "Random Forest": "Random_Forest_model_avec_duration_sans_parametres.pkl",
        "Logistic Regression": "Logistic_Regression_model_avec_duration_sans_parametres.pkl",
        "Decision Tree": "Decision_Tree_model_avec_duration_sans_parametres.pkl",
        "KNN": "KNN_model_avec_duration_sans_parametres.pkl",
        "AdaBoost": "AdaBoost_model_avec_duration_sans_parametres.pkl",
        "Bagging": "Bagging_model_avec_duration_sans_parametres.pkl",
        "SVM": "SVM_model_avec_duration_sans_parametres.pkl",
        "XGBOOST": "XGBOOST_model_avec_duration_sans_parametres.pkl",
    }

        
    # Résultats des modèles
    results_sans_param = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        # Align X_test columns to model's expected columns
        if hasattr(trained_clf, 'feature_names_in_'):
            expected_cols = trained_clf.feature_names_in_
            for col in expected_cols:
                if col not in X_test.columns:
                    X_test[col] = 0  # or np.nan, depending on your model's needs
            X_test_aligned = X_test[expected_cols]
        else:
            X_test_aligned = X_test
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test_aligned)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_sans_param[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    results_sans_param = pd.DataFrame(results_sans_param).T
    results_sans_param.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    results_sans_param = results_sans_param.sort_values(by="Recall", ascending=False)

    # Graphiques
    results_melted = results_sans_param.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    results_melted.rename(columns={"index": "Classifier"}, inplace=True)

    #HYPERPARAMÈTRES : ÉTAPE 1 - RECHERCHES TEAM POUR LES 3 TOPS MODÈLES (RANDOM FOREST / SVM / XGBOOST)
    # Initialisation des classifiers
    #classifiers_AD_TEAM = {
    #"RF_dounia": RandomForestClassifier(max_depth=None, max_features='log2',min_samples_leaf=2, min_samples_split=2, n_estimators=200, random_state=42),
    #"RF_dilene": RandomForestClassifier(class_weight='balanced', max_depth=25, max_features='sqrt',min_samples_leaf=1, min_samples_split=15, n_estimators=1500, random_state=42),
    #"RF_fatou": RandomForestClassifier(max_depth= None,max_features='log2',min_samples_leaf=2,min_samples_split=2,n_estimators=200,random_state=42),
    #"RF_carolle": RandomForestClassifier(class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42),
    #"SVM_dounia": svm.SVC(C = 1, class_weight = "balanced", gamma = 'scale', kernel = 'rbf', random_state=42),
    #"SVM_dilene": svm.SVC(C=0.1, class_weight='balanced', gamma=0.1, kernel='rbf', random_state=42),
    #"SVM_fatou": svm.SVC(kernel='rbf',gamma='scale', C=1, random_state=42),
    #"SVM_carolle": svm.SVC(C=0.1, class_weight='balanced', gamma='scale', kernel='rbf', random_state=42),
    #"XGBOOST_dounia": XGBClassifier(colsample_bytree=1.0, learning_rate=0.05,max_depth=7,min_child_weight=1,n_estimators=200,subsample=0.8,random_state=42),
    #"XGBOOST_dilene": XGBClassifier(base_score=0.3, gamma=14, learning_rate=0.6, max_delta_step=1, max_depth=27,min_child_weight=2, n_estimators=900,random_state=42),
    #"XGBOOST_carolle": XGBClassifier(colsample_bytree=0.8, gamma=10, max_depth=17,min_child_weight=1,n_estimators=1000, reg_lambda=0.89, random_state=42),
    #"XGBOOST_fatou": XGBClassifier(colsample_bytree=0.8, gamma= 5, learning_rate= 0.1, max_depth= 5, n_estimators= 100, subsample= 0.8, random_state=42)
    #}

    # Résultats des modèles
    #results_AD_top_3_hyperparam_TEAM = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_team(model_name, clf, X_train, y_train):
        #filename = f"{model_name.replace(' ', '_')}_model_AD_TOP_3_hyperparam_TEAM.pkl"  # Nom du fichier
        #try:
            # Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    # Boucle pour entraîner ou charger les modèles
    #for name, clf in classifiers_AD_TEAM.items():
        # Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_team(name, clf, X_train, y_train)
        #y_pred = trained_clf.predict(X_test)
            
        # Calculer les métriques
        #accuracy = accuracy_score(y_test, y_pred)
        #f1 = f1_score(y_test, y_pred)
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        #results_AD_top_3_hyperparam_TEAM[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall,
        #}
    #COMME ON A ENREGISTRÉ LES MODÈLES, VOICI LE NOUVEAU CODE À UTILISER : 
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_team = {
        "RF_dounia": "RF_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "RF_fatou": "RF_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "RF_carolle": "RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_dounia": "SVM_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_dilene": "SVM_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_fatou": "SVM_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM_carolle": "SVM_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_dounia": "XGBOOST_dounia_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_dilene": "XGBOOST_dilene_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_carolle": "XGBOOST_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST_fatou": "XGBOOST_fatou_model_AD_TOP_3_hyperparam_TEAM.pkl",
    }


    # Résultats des modèles
    results_AD_top_3_hyperparam_TEAM = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_team.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test_aligned)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_AD_top_3_hyperparam_TEAM[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    df_results_AD_top_3_hyperparam_TEAM = pd.DataFrame(results_AD_top_3_hyperparam_TEAM).T
    df_results_AD_top_3_hyperparam_TEAM.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_AD_top_3_hyperparam_TEAM = df_results_AD_top_3_hyperparam_TEAM.sort_values(by="Recall", ascending=False)
    
    melted_df_results_AD_top_3_hyperparam_TEAM = df_results_AD_top_3_hyperparam_TEAM.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_AD_top_3_hyperparam_TEAM.rename(columns={"index": "Classifier"}, inplace=True)    


    #HYPERPARAMÈTRES : ÉTAPE 2 - 
    # Initialisation des classifiers
    #classifiers_grid_2 = {
    #"Random Forest GridSearch2": RandomForestClassifier(class_weight= 'balanced', max_depth = None, max_features = 'sqrt', min_samples_leaf= 2, min_samples_split= 15, n_estimators = 200, random_state=42),
    #"SVM GridSearch2": svm.SVC (C = 1, class_weight = 'balanced', gamma = 'scale', kernel ='rbf', random_state=42),
    #"XGBOOST GridSearch2": XGBClassifier (colsample_bytree = 0.8, gamma = 5, learning_rate = 0.05, max_depth = 17, min_child_weight = 1, n_estimators = 200, subsample = 0.8, random_state=42)
    #}

    # Résultats des modèles
    #results_hyperparam_gridsearch2 = {}

    #Fonction pour entraîner et sauvegarder un modèle
    #def train_and_save_model_team(model_name, clf, X_train, y_train):
        #filename = f"{model_name.replace(' ', '_')}_model_AD_TOP_3_hyperparam_TEAM.pkl"  # Nom du fichier
        #try:
            # Charger le modèle si le fichier existe déjà
            #trained_clf = joblib.load(filename)
        #except FileNotFoundError:
            # Entraîner et sauvegarder le modèle
            #clf.fit(X_train, y_train)
            #joblib.dump(clf, filename)
            #trained_clf = clf
        #return trained_clf

    # Boucle pour entraîner ou charger les modèles
    #for name, clf in classifiers_grid_2.items():
        # Entraîner ou charger le modèle
        #trained_clf = train_and_save_model_team(name, clf, X_train, y_train)
        #y_pred = trained_clf.predict(X_test)
            
        # Calculer les métriques
        #accuracy = accuracy_score(y_test, y_pred)
        #f1 = f1_score(y_test, y_pred)
        #precision = precision_score(y_test, y_pred)
        #recall = recall_score(y_test, y_pred)
            
        # Stocker les résultats
        #results_hyperparam_gridsearch2[name] = {
            #"Accuracy": accuracy,
            #"F1 Score": f1,
            #"Precision": precision,
            #"Recall": recall
        #}
    
    #LES MODÈLES PRÉCÉDENTS ONT ÉTÉ ENREGISTRÉS VIA JOBLIB donc nouveau code pour appeler ces modèles enregistrés
    # Liste des modèles enregistrés et leurs fichiers correspondants
    model_files_grid_2 = {
        "Random Forest GridSearch2": "Random_Forest_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "SVM GridSearch2": "SVM_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
        "XGBOOST GridSearch2": "XGBOOST_GridSearch2_model_AD_TOP_3_hyperparam_TEAM.pkl",
    }

    # Résultats des modèles
    results_hyperparam_gridsearch2 = {}

    # Boucle pour charger les modèles et calculer les métriques
    for name, file_path in model_files_grid_2.items():
        # Charger le modèle sauvegardé
        trained_clf = joblib.load(file_path)
        
        # Faire des prédictions
        y_pred = trained_clf.predict(X_test_aligned)

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Stocker les résultats
        results_hyperparam_gridsearch2[name] = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
        }

    # Conversion des résultats en DataFrame
    df_results_hyperparam_gridsearch2 = pd.DataFrame(results_hyperparam_gridsearch2).T
    df_results_hyperparam_gridsearch2.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    df_results_hyperparam_gridsearch2 = df_results_hyperparam_gridsearch2.sort_values(by="Recall", ascending=False)
    
    melted_df_results_hyperparam_gridsearch2 = df_results_hyperparam_gridsearch2.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    melted_df_results_hyperparam_gridsearch2.rename(columns={"index": "Classifier"}, inplace=True)    



    if page == pages[0] : 
        st.subheader("Méthodologie")
        st.write("Nous avons effectué **deux modélisations**, l'une en **conservant la variable Duration** et **l'autre sans la variable Duration**: étant donné que cette variable **ne peut être connue qu'après le contact avec le client**.")
        st.write("Pour chaque modélisation, avec ou sans Duration, nous avons analysé les scores des principaux modèles de classification d'abord **sans paramètres** afin de sélectionner les 3 meilleurs modèles, **puis sur ces 3 modèles nous avons effectué des recherches d'hyperparamètres** à l'aide de la **fonction GridSearchCV** afin de sélectionner le modèle **le plus performant possible.**")
        st.write("Enfin sur le meilleur modèle trouvé, nous avons effectué une **analyse SHAP afin d'interpréter les décisions prises par le modèle** dans la détection des clients susceptibles de Deposit YES.")
        st.write("La **métrique principale** choisie est le **Recall pour la classe 1 (deposit = 1)**, afin d'**optimiser la détection des clients intéressés par le DAT en réduisant les faux négatifs**. ")
        st.write("L'objectif de notre modélisation est de **maximiser la performance selon cette métrique.** ")
                 
    if page == pages[1] : 
        #AVEC DURATION
        submenu_modelisation = st.radio("", ("Scores modèles sans paramètres", "Hyperparamètres et choix du modèle"), horizontal=True)
        if submenu_modelisation == "Scores modèles sans paramètres" :
            st.subheader("Scores modèles sans paramètres")
            st.write("Tableau avec les résultats des modèles :")
            st.dataframe(results_sans_param)
                
            st.write("Visualisation des différents scores :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=results_melted,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
                
    
        if submenu_modelisation == "Hyperparamètres et choix du modèle" :
            st.subheader("Hyperparamètres et choix du modèle")
            st.write("")
            
            st.subheader("Étape 1 : Team GridSearch top 3 modèles")
            st.write("Recherches Gridsearch des 4 membres de la Team sur les top 3 modèles ressortis sans paramètres")
            st.write("Tableau des résultats des modèles hyperparamétrés")
            st.dataframe(df_results_AD_top_3_hyperparam_TEAM)
          
            
            st.subheader("Étape 2 : Modèle sélectionné")
            st.write("Le modèle Random Forest 'RF_carolle' avec les hyperparamètres ci-dessous affiche la meilleure performance en termes de Recall, aussi nous choisisons de poursuivre notre modélisation avec ce modèle")
            st.write("RandomForestClassifier(**class_weight= 'balanced', max_depth=20, max_features='sqrt',min_samples_leaf=2, min_samples_split=10, n_estimators= 200, random_state=42**)")
                
            # Chargement du modèle enregistré
            filename = "RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl"
            rf_carolle_model = joblib.load(filename)

            # Prédictions sur les données test
            y_pred = rf_carolle_model.predict(X_test_aligned)

            # Calcul des métriques pour chaque classe
            report = classification_report(y_test, y_pred, target_names=["Classe 0", "Classe 1"], output_dict=True)

            # Conversion du rapport en DataFrame pour affichage en tableau
            report_df = pd.DataFrame(report).T

            # Arrondi des valeurs à 4 décimales pour un affichage propre
            report_df = report_df.round(4)

            # Suppression des colonnes inutiles si besoin
            report_df = report_df.drop(columns=["support"])

            # Affichage global du rapport sous forme de tableau
            st.write("**Rapport de classification du modèle:**")
            st.table(report_df)

            # Création de la matrice de confusion sous forme de DataFrame
            st.write("**Matrice de confusion du modèle:**")
            table_rf = pd.crosstab(y_test, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_rf)

            # Création de la matrice de confusion sous forme de DataFrame
            st.write("**Matrice de confusion du modèle :**")
            table_xgboost_1 = pd.crosstab(y_test, y_pred, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_xgboost_1)

            # Affichage global du rapport sous forme de tableau
            st.write("**Rapport de classification du modèle :**")
            st.table(report_df)
            
            st.subheader("Bar plot - Importance des variables")
            st.write("Graphique d'importance des variables pour le modèle avec Duration")
            
            try:
                # Load SHAP values for the model with duration
                shap_values_RF_carolle = joblib.load("shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
                
                # Calculate feature importance using SHAP values
                shap_abs = np.abs(shap_values_RF_carolle.values).mean(axis=0)
                if shap_abs.ndim > 1:
                    shap_abs = shap_abs.flatten()
                
                # Load the model to get feature names
                model_RF_carolle = joblib.load("RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
                X_test_aligned = align_X_test(X_test, model_RF_carolle)
                feature_names = X_test_aligned.columns
                
                # Ensure both arrays have the same length
                min_length = min(len(feature_names), len(shap_abs))
                feature_names = feature_names[:min_length]
                shap_abs = shap_abs[:min_length]
                
                # Create DataFrame and sort by importance
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_abs})
                importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                
                # Create custom bar plot
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(range(len(importance_df)), importance_df['Importance'])
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df['Feature'])
                ax.set_xlabel('Mean |SHAP|')
                ax.set_title('Top 10 Most Important Features (with Duration)')
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{width:.4f}', ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
                
                st.write("**Interprétation :**")
                st.write("- Les variables les plus importantes pour prédire la souscription avec Duration sont :")
                for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                    st.write(f"  {i}. **{row['Feature']}** : {row['Importance']:.4f}")
                
            except FileNotFoundError:
                st.warning("Les valeurs SHAP pour le modèle avec Duration ne sont pas disponibles.")
                st.write("Pour générer ces valeurs, il faudrait :")
                st.write("1. Charger le modèle Random Forest optimisé")
                st.write("2. Créer l'explainer SHAP")
                st.write("3. Calculer les SHAP values")
                st.write("4. Sauvegarder les résultats")

    if page == pages[2] :
        #SANS DURATION
        submenu_modelisation2 = st.radio(" ", ("Scores sans paramètres", "Hyperparamètres et choix du modèle"), horizontal = True)
    
        if submenu_modelisation2 == "Scores sans paramètres" :
            st.subheader("Scores des modèles sans paramètres")
            
            #RÉSULTAT DES MODÈLES SANS PARAMETRES (CODE UTILISÉ UNE FOIS POUR CHARGER LES MODÈLES)
            # Initialisation des classifiers
            #classifiers_SD= {
                #"Random Forest": RandomForestClassifier(random_state=42),
                #"Logistic Regression": LogisticRegression(random_state=42),
                #"Decision Tree": DecisionTreeClassifier(random_state=42),
                #"KNN": KNeighborsClassifier(),
                #"AdaBoost": AdaBoostClassifier(random_state=42),
                #"Bagging": BaggingClassifier(random_state=42),
                #"SVM": svm.SVC(random_state=42),
                #"XGBOOST": XGBClassifier(random_state=42),
            #}

            # Résultats des modèles
            #results_SD_sans_param = {}

            #Fonction pour entraîner et sauvegarder un modèle
            #def train_and_save_model_SD_sans_param(model_name, clf, X_train_sd, y_train_sd):
                #filename = f"{model_name.replace(' ', '_')}_model_sans_duration_sans_parametres.pkl"  # Nom du fichier
                #try:
                    # Charger le modèle si le fichier existe déjà
                    #trained_clf = joblib.load(filename)
                #except FileNotFoundError:
                    # Entraîner et sauvegarder le modèle
                    #clf.fit(X_train_sd, y_train_sd)
                    #joblib.dump(clf, filename)
                    #trained_clf = clf
                #return trained_clf

            # Boucle pour entraîner ou charger les modèles
            #for name, clf in classifiers_SD.items():
                # Entraîner ou charger le modèle
                #trained_clf = train_and_save_model_SD_sans_param(name, clf, X_train_sd, y_train_sd)
                #y_pred = trained_clf.predict(X_test_sd)
                    
                # Calculer les métriques
                #accuracy = accuracy_score(y_test_sd, y_pred)
                #f1 = f1_score(y_test_sd, y_pred)
                #precision = precision_score(y_test_sd, y_pred)
                #recall = recall_score(y_test_sd, y_pred)
                    
                # Stocker les résultats
                #results_SD_sans_param[name] = {
                    #"Accuracy": accuracy,
                    #"F1 Score": f1,
                    #"Precision": precision,
                    #"Recall": recall
                #}

            #CODE À UTILISER PUISQUE MODÈLES SAUVEGARDÉS
            #Chargement des modèles préalablement enregistrés
            models_SD = {
                "Random Forest": joblib.load("Random_Forest_model_sans_duration_sans_parametres.pkl"),
                "Logistic Regression": joblib.load("Logistic_Regression_model_sans_duration_sans_parametres.pkl"),
                "Decision Tree": joblib.load("Decision_Tree_model_sans_duration_sans_parametres.pkl"),
                "KNN": joblib.load("KNN_model_sans_duration_sans_parametres.pkl"),
                "AdaBoost": joblib.load("AdaBoost_model_sans_duration_sans_parametres.pkl"),
                "Bagging": joblib.load("Bagging_model_sans_duration_sans_parametres.pkl"),
                "SVM": joblib.load("SVM_model_sans_duration_sans_parametres.pkl"),
                "XGBOOST": joblib.load("XGBOOST_model_sans_duration_sans_parametres.pkl")
            }
            # Charger votre modèle
            filename = "Random_Forest_model_sans_duration_sans_parametres.pkl"
            model = joblib.load(filename)

            # Sauvegarder le modèle avec compression de niveau 9
            joblib.dump(model, "Random_Forest_model_sans_duration_sans_parametres.pkl", compress=5)
    
            # Résultats des modèles
            results_SD_sans_param = {}

            # Boucle pour charger les modèles et calculer les résultats
            for name, trained_clf in models_SD.items():
                # Align X_test_sd to model's expected columns
                if hasattr(trained_clf, 'feature_names_in_'):
                    expected_cols = trained_clf.feature_names_in_
                    for col in expected_cols:
                        if col not in X_test_sd.columns:
                            X_test_sd[col] = 0
                    X_test_sd_aligned = X_test_sd[expected_cols]
                else:
                    X_test_sd_aligned = X_test_sd
                # Prédictions sur les données test
                y_pred = trained_clf.predict(X_test_sd_aligned)

                # Calculer les métriques
                accuracy = accuracy_score(y_test_sd, y_pred)
                f1 = f1_score(y_test_sd, y_pred)
                precision = precision_score(y_test_sd, y_pred)
                recall = recall_score(y_test_sd, y_pred)

                # Stocker les résultats
                results_SD_sans_param[name] = {
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall
                }

            # Conversion des résultats en DataFrame
            df_results_SD_sans_param = pd.DataFrame(results_SD_sans_param).T
            df_results_SD_sans_param.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            df_results_SD_sans_param = df_results_SD_sans_param.sort_values(by="Recall", ascending=False)
            
            melted_df_results_SD_sans_param = df_results_SD_sans_param.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
            melted_df_results_SD_sans_param.rename(columns={"index": "Classifier"}, inplace=True)
            
            st.write("La variable **'duration'** a été **retirée** du dataset, les modèles ont été testés sans paramètres et classés selon le score 'Recall' afin de sélectionner les tops modèles pour optimisation ultérieure.")
            st.dataframe(df_results_SD_sans_param)
            
            st.write("Visualisation des résultats :")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_df_results_SD_sans_param,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("Ces scores nous permettent de sélectionner notre **top 3 des modèles** à tester avec le GridSearchCV :  \n\
            1. Le modèle **Decision Tree**  \n\
            2. Le modèle **XGBOOST**  \n\
            3. Le modèle **Random Forest**")

            st.markdown("Puisque le modèle **SVM** affiche un **meilleur résultat sur le score de Précision**, nous allons également effectuer des tests avec ce modèle, en plus des 3 modèles listés ci-dessus.")

        if submenu_modelisation2 == "Hyperparamètres et choix du modèle" :
            st.write("Scores des modèles hyperparamétrés :")            
            #CODE CHARGÉ UNE FOIS POUR LOAD PUIS RETIRÉ
            # Initialisation des classifiers
            #classifiers_SD_hyperparam= {
                #"Random Forest": RandomForestClassifier(class_weight='balanced', max_depth=8,  max_features='log2', min_samples_leaf=250, min_samples_split=300, n_estimators=400, random_state=42),
                #"Decision Tree": DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=5,  max_features=None, min_samples_leaf=100, min_samples_split=2, random_state=42),
                #"SVM" : svm.SVC(C=0.01, class_weight='balanced', gamma='scale', kernel='linear',random_state=42),
                #"XGBOOST_1" : XGBClassifier(gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42),
                #"XGBOOST_2" : XGBClassifier(gamma=0.05,colsample_bytree=0.88, learning_rate=0.39, max_depth=6, min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.8, scale_pos_weight=2.56, subsample=0.99, random_state=42),
                #"XGBOOST_3" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
                #"XGBOOST_TESTDIL" : XGBClassifier(gamma=0.05,colsample_bytree=0.83, learning_rate=0.37, max_depth=6,  min_child_weight=1.2, n_estimators=30, reg_alpha=1.2, reg_lambda=1.7, scale_pos_weight=2.46, subsample=0.99, random_state=42),
            #}

            # Résultats des modèles
            #results_SD_TOP_4_hyperparam = {}

            #Fonction pour entraîner et sauvegarder un modèle
            #def train_and_save_model_SD_hyperparam(model_name, clf, X_train_sd, y_train_sd):
                #filename = f"{model_name.replace(' ', '_')}_model_SD_TOP_4_hyperparam.pkl"  # Nom du fichier
                #try:
                    # Charger le modèle si le fichier existe déjà
                    #trained_clf = joblib.load(filename)
                #except FileNotFoundError:
                    # Entraîner et sauvegarder le modèle
                    #clf.fit(X_train_sd, y_train_sd)
                    #joblib.dump(clf, filename)
                    #trained_clf = clf
                #return trained_clf

            # Boucle pour entraîner ou charger les modèles
            #for name, clf in classifiers_SD_hyperparam.items():
                # Entraîner ou charger le modèle
                #trained_clf = train_and_save_model_SD_hyperparam(name, clf, X_train_sd, y_train_sd)
                #y_pred = trained_clf.predict(X_test_sd)
                    
                # Calculer les métriques
                #accuracy = accuracy_score(y_test_sd, y_pred)
                #f1 = f1_score(y_test_sd, y_pred)
                #precision = precision_score(y_test_sd, y_pred)
                #recall = recall_score(y_test_sd, y_pred)
                    
                # Stocker les résultats
                #results_SD_TOP_4_hyperparam[name] = {
                    #"Accuracy": accuracy,
                    #"F1 Score": f1,
                    #"Precision": precision,
                    #"Recall": recall
                #}
            
            #Chargement des modèles préalablement enregistrés
            models_SD_hyperparam = {
                "Random Forest": joblib.load("Random_Forest_model_SD_TOP_4_hyperparam.pkl"),
                "Decision Tree": joblib.load("Decision_Tree_model_SD_TOP_4_hyperparam.pkl"),
                "SVM": joblib.load("SVM_model_SD_TOP_4_hyperparam.pkl"),
                "XGBOOST": joblib.load("XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"),
            }

            # Résultats des modèles
            results_SD_TOP_4_hyperparam = {}

            # Boucle pour charger les modèles et calculer les résultats
            for name, trained_clf in models_SD_hyperparam.items():
                # Align X_test_sd to model's expected columns
                if hasattr(trained_clf, 'feature_names_in_'):
                    expected_cols = trained_clf.feature_names_in_
                    for col in expected_cols:
                        if col not in X_test_sd.columns:
                            X_test_sd[col] = 0
                    X_test_sd_aligned = X_test_sd[expected_cols]
                else:
                    X_test_sd_aligned = X_test_sd
                # Prédictions sur les données test
                y_pred = trained_clf.predict(X_test_sd_aligned)

                # Calculer les métriques
                accuracy = accuracy_score(y_test_sd, y_pred)
                f1 = f1_score(y_test_sd, y_pred)
                precision = precision_score(y_test_sd, y_pred)
                recall = recall_score(y_test_sd, y_pred)

                # Stocker les résultats
                results_SD_TOP_4_hyperparam[name] = {
                    "Accuracy": accuracy,
                    "F1 Score": f1,
                    "Precision": precision,
                    "Recall": recall
                }
            
            # Conversion des résultats en DataFrame
            df_results_SD_TOP_4_hyperparam = pd.DataFrame(results_SD_TOP_4_hyperparam).T
            df_results_SD_TOP_4_hyperparam.columns = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            df_results_SD_TOP_4_hyperparam = df_results_SD_TOP_4_hyperparam.sort_values(by="Recall", ascending=False)
            
            melted_df_results_SD_TOP_4_hyperparam = df_results_SD_TOP_4_hyperparam.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
            melted_df_results_SD_TOP_4_hyperparam.rename(columns={"index": "Classifier"}, inplace=True)
            
            st.dataframe(df_results_SD_TOP_4_hyperparam)
            
            st.write("Visualisation des résultats:")
            # Visualisation des résultats des différents modèles :
            fig = plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_df_results_SD_TOP_4_hyperparam,x="Classifier",y="Score",hue="Metric",palette="rainbow")
            # Ajouter des titres et légendes
            plt.title("Performance des modèles par métrique", fontsize=16)
            plt.xlabel("Modèles", fontsize=14)
            plt.ylabel("Scores", fontsize=14)
            plt.xticks(rotation=45)
            plt.legend(title="Métrique", fontsize=12)
            plt.legend(loc='lower right')
            plt.tight_layout()
            st.pyplot(fig) 

            st.markdown("Étant donné les scores obtenus sur ces modèles avec hyperparamètres, **nous retenons le modèle XGBOOST** qui affiche un bien meilleur Recall Score sur la classe 1.")
     
                    
            st.subheader("Modèle sélectionné")
            st.write("Voici les hyperparamètres du modèle XGBOOST retenu :")
            st.write("XGBClassifier(**gamma=0.05,colsample_bytree=0.9, learning_rate=0.39, max_depth=6, min_child_weight=1.29, n_estimators=34, reg_alpha=1.29, reg_lambda=1.9, scale_pos_weight=2.6, subsample=0.99, random_state=42**)")
                
            # Chargement du modèle enregistré
            filename = "XGBOOST_1_model_SD_TOP_4_hyperparam.pkl"
            model_XGBOOST_1_model_SD_TOP_4_hyperparam = joblib.load(filename)

            # Align X_test_sd to model's expected columns
            if hasattr(model_XGBOOST_1_model_SD_TOP_4_hyperparam, 'feature_names_in_'):
                expected_cols = model_XGBOOST_1_model_SD_TOP_4_hyperparam.feature_names_in_
                for col in expected_cols:
                    if col not in X_test_sd.columns:
                        X_test_sd[col] = 0
                X_test_sd_aligned = X_test_sd[expected_cols]
            else:
                X_test_sd_aligned = X_test_sd
            # Prédictions sur les données test
            y_pred_1 = model_XGBOOST_1_model_SD_TOP_4_hyperparam.predict(X_test_sd_aligned)

            # Calcul des métriques pour chaque classe
            report_1 = classification_report(y_test_sd, y_pred_1, target_names=["Classe 0", "Classe 1"], output_dict=True)

            # Conversion du rapport en DataFrame pour affichage en tableau
            report_df_1 = pd.DataFrame(report_1).T

            # Arrondi des valeurs à 4 décimales pour un affichage propre
            report_df_1 = report_df_1.round(4)

            # Suppression des colonnes inutiles si besoin
            report_df_1 = report_df_1.drop(columns=["support"])


            # Création de la matrice de confusion sous forme de DataFrame
            st.write("**Matrice de confusion du modèle :**")
            table_xgboost_1 = pd.crosstab(y_test_sd, y_pred_1, rownames=["Réalité"], colnames=["Prédiction"])
            st.dataframe(table_xgboost_1)

            # Affichage global du rapport sous forme de tableau
            st.write("**Rapport de classification du modèle :**")
            st.table(report_df_1)
            


        
if selected == 'Interprétation':      
    st.sidebar.title("SOUS MENU INTERPRÉTATION")
    pages=["INTERPRÉTATION AVEC DURATION", "INTERPRÉTATION SANS DURATION"]
    page=st.sidebar.radio('AVEC ou SANS Duration', pages)

    if page == pages[0] : 
        st.subheader("Interprétation SHAP avec la colonne Duration")
        #submenu_interpretation = st.selectbox("Menu", ("Summary plot", "Bar plot poids des variables", "Analyses des variables catégorielles", "Dependence plots"))
        submenu_interpretation_Duration = st.radio("", ("ANALYSE GLOBALE", "ANALYSE DES VARIABLES LES PLUS INFLUENTES"), horizontal=True)


        if submenu_interpretation_Duration == "ANALYSE GLOBALE" :
            submenu_globale = st.radio("", ("Summary plot", "Bar plot"), horizontal=True) 

            if submenu_globale == "Summary plot" :
                st.subheader("Summary plot")
                # Affichage des visualisations SHAP
                #SHAP
                #PARTIE DU CODE À VIRER UNE FOIS LES SHAP VALUES CHARGÉES
                #Chargement du modèle XGBOOST_1 déjà enregistré
                #filename_RF_carolle = "RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl"
                #model_RF_carolle_model_AD_TOP_3_hyperparam_TEAM = joblib.load(filename_RF_carolle)

                #Chargement des données pour shap 
                #data_to_explain_RF_carolle = X_test  

                #Création de l'explainer SHAP pour XGBOOST_1
                #explainer_RF_carolle = shap.TreeExplainer(model_RF_carolle_model_AD_TOP_3_hyperparam_TEAM)

                #Calcul des shap values
                #shap_values_RF_carolle = explainer_RF_carolle(data_to_explain_RF_carolle)

                #Sauvegarder des shap values avec joblib
                #joblib.dump(shap_values_RF_carolle, "shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                #CODE À UTILISER UNE FOIS LES SHAP VALUES CHARGÉES
                shap_values_RF_carolle = joblib.load("shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")

                # Load the model to get expected columns
                model_RF_carolle = joblib.load("RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
                X_test_aligned = align_X_test(X_test, model_RF_carolle)

                shap.summary_plot(shap_values_RF_carolle, X_test_aligned, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            elif submenu_globale == "Bar plot" :
                st.subheader("Bar plot - Importance des variables")
                st.write("Graphique d'importance des variables pour le modèle avec Duration")
                
                try:
                    # Load SHAP values for the model with duration
                    shap_values_RF_carolle = joblib.load("shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
                    
                    shap.summary_plot(shap_values_RF_carolle, plot_type="bar", show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    
                    st.write("**Interprétation :**")
                    st.write("- Les variables les plus importantes pour prédire la souscription avec Duration sont :")
                    st.write("  1. **duration** : La durée de l'appel (variable la plus importante)")
                    st.write("  2. **balance** : Le solde du compte client")
                    st.write("  3. **age** : L'âge du client") 
                    st.write("  4. **campaign** : Le nombre de contacts pendant la campagne")
                    st.write("  5. **previous** : Le nombre de contacts précédents")
                    
                except FileNotFoundError:
                    st.warning("Les valeurs SHAP pour le modèle avec Duration ne sont pas disponibles.")
                    st.write("Pour générer ces valeurs, il faudrait :")
                    st.write("1. Charger le modèle Random Forest optimisé")
                    st.write("2. Créer l'explainer SHAP")
                    st.write("3. Calculer les SHAP values")
                    st.write("4. Sauvegarder les résultats")

        elif submenu_interpretation_Duration == "ANALYSE DES VARIABLES LES PLUS INFLUENTES":
            st.subheader("Variables les plus influentes (SHAP)")
            shap_values_RF_carolle = joblib.load("shap_values_RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
            model_RF_carolle = joblib.load("RF_carolle_model_AD_TOP_3_hyperparam_TEAM.pkl")
            X_test_aligned = align_X_test(X_test, model_RF_carolle)
            # Get mean absolute SHAP values for each feature (with Duration)
            shap_abs = np.abs(shap_values_RF_carolle.values).mean(axis=0)
            if shap_abs.ndim > 1:
                shap_abs = shap_abs.flatten()
            feature_names = X_test_aligned.columns
            
            # Ensure both arrays have the same length
            min_length = min(len(feature_names), len(shap_abs))
            feature_names = feature_names[:min_length]
            shap_abs = shap_abs[:min_length]
            
            top_features = pd.DataFrame({'Feature': feature_names, 'Mean |SHAP|': shap_abs})
            top_features = top_features.sort_values('Mean |SHAP|', ascending=False).head(10)
            st.dataframe(top_features)
            # Optionally, add a dependence plot for the top feature
            # shap.dependence_plot(top_features.iloc[0]['Feature'], shap_values_RF_carolle.values, X_test_aligned, show=False)
            # st.pyplot(plt.gcf())
            # plt.clf()
            
            # Show a simple bar chart instead
            st.subheader("Top 10 Most Influential Features")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = top_features.head(10)
            ax.barh(range(len(top_10)), top_10['Mean |SHAP|'])
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['Feature'])
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title('Top 10 Most Influential Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

    if page == pages[1] : 
        st.subheader("Interprétation SHAP sans la colonne Duration")
        submenu_interpretation_SansDuration = st.radio("", ("ANALYSE GLOBALE", "ANALYSE DES VARIABLES LES PLUS INFLUENTES"), horizontal=True)

        if submenu_interpretation_SansDuration == "ANALYSE GLOBALE" :
            submenu_globale_sd = st.radio("", ("Summary plot", "Bar plot"), horizontal=True) 

            if submenu_globale_sd == "Summary plot" :
                st.subheader("Summary plot - Modèle sans Duration")
                st.write("Analyse SHAP du modèle XGBOOST optimisé sans la variable Duration")
                try:
                    shap_values_XGBOOST_sd = joblib.load("shap_values_XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
                    model_XGBOOST_sd = joblib.load("XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
                    X_test_sd_aligned = align_X_test(X_test_sd, model_XGBOOST_sd)
                    shap.summary_plot(shap_values_XGBOOST_sd, X_test_sd_aligned, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                    st.write("**Interprétation :**")
                    st.write("- Les variables les plus importantes pour prédire la souscription sans Duration sont :")
                    st.write("  1. **balance** : Le solde du compte client")
                    st.write("  2. **age** : L'âge du client") 
                    st.write("  3. **campaign** : Le nombre de contacts pendant la campagne")
                    st.write("  4. **previous** : Le nombre de contacts précédents")
                    st.write("  5. **job** : Le type d'emploi du client")
                except FileNotFoundError:
                    st.warning("Les valeurs SHAP pour le modèle sans Duration ne sont pas encore calculées.")
                    st.write("Pour générer ces valeurs, il faudrait :")
                    st.write("1. Charger le modèle XGBOOST optimisé")
                    st.write("2. Créer l'explainer SHAP")
                    st.write("3. Calculer les SHAP values")
                    st.write("4. Sauvegarder les résultats")

            elif submenu_globale_sd == "Bar plot" :
                st.subheader("Bar plot - Importance des variables")
                st.write("Graphique d'importance des variables pour le modèle sans Duration")
                try:
                    shap_values_XGBOOST_sd = joblib.load("shap_values_XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
                    
                    # Calculate feature importance using SHAP values
                    shap_abs = np.abs(shap_values_XGBOOST_sd.values).mean(axis=0)
                    if shap_abs.ndim > 1:
                        shap_abs = shap_abs.flatten()
                    
                    # Load the model to get feature names
                    model_XGBOOST_sd = joblib.load("XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
                    X_test_sd_aligned = align_X_test(X_test_sd, model_XGBOOST_sd)
                    feature_names = X_test_sd_aligned.columns
                    
                    # Ensure both arrays have the same length
                    min_length = min(len(feature_names), len(shap_abs))
                    feature_names = feature_names[:min_length]
                    shap_abs = shap_abs[:min_length]
                    
                    # Create DataFrame and sort by importance
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': shap_abs})
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                    
                    # Create custom bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(range(len(importance_df)), importance_df['Importance'])
                    ax.set_yticks(range(len(importance_df)))
                    ax.set_yticklabels(importance_df['Feature'])
                    ax.set_xlabel('Mean |SHAP|')
                    ax.set_title('Top 10 Most Important Features (without Duration)')
                    
                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{width:.4f}', ha='left', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.clf()
                    
                    st.write("**Interprétation :**")
                    st.write("- Les variables les plus importantes pour prédire la souscription sans Duration sont :")
                    for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                        st.write(f"  {i}. **{row['Feature']}** : {row['Importance']:.4f}")
                        
                except FileNotFoundError:
                    st.warning("Les valeurs SHAP pour le modèle sans Duration ne sont pas encore calculées.")

        elif submenu_interpretation_SansDuration == "ANALYSE DES VARIABLES LES PLUS INFLUENTES":
            st.subheader("Variables les plus influentes (SHAP)")
            shap_values_XGBOOST_sd = joblib.load("shap_values_XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
            model_XGBOOST_sd = joblib.load("XGBOOST_1_model_SD_TOP_4_hyperparam.pkl")
            X_test_sd_aligned = align_X_test(X_test_sd, model_XGBOOST_sd)
            # Get mean absolute SHAP values for each feature (sans Duration)
            shap_abs = np.abs(shap_values_XGBOOST_sd.values).mean(axis=0)
            if shap_abs.ndim > 1:
                shap_abs = shap_abs.flatten()
            feature_names = X_test_sd_aligned.columns
            
            # Ensure both arrays have the same length
            min_length = min(len(feature_names), len(shap_abs))
            feature_names = feature_names[:min_length]
            shap_abs = shap_abs[:min_length]
            
            top_features = pd.DataFrame({'Feature': feature_names, 'Mean |SHAP|': shap_abs})
            top_features = top_features.sort_values('Mean |SHAP|', ascending=False).head(10)
            st.dataframe(top_features)
            # Optionally, add a dependence plot for the top feature
            # shap.dependence_plot(top_features.iloc[0]['Feature'], shap_values_XGBOOST_sd.values, X_test_sd_aligned, show=False)
            # st.pyplot(plt.gcf())
            # plt.clf()
            
            # Show a simple bar chart instead
            st.subheader("Top 10 Most Influential Features")
            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = top_features.head(10)
            ax.barh(range(len(top_10)), top_10['Mean |SHAP|'])
            ax.set_yticks(range(len(top_10)))
            ax.set_yticklabels(top_10['Feature'])
            ax.set_xlabel('Mean |SHAP|')
            ax.set_title('Top 10 Most Influential Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()

if selected == 'Recommandations & Perspectives':
    st.title("Recommandations & Perspectives")
    st.write("""
    ### Recommandations
    - **Améliorer la collecte de données** : Ajouter plus de variables comportementales pour mieux cibler les clients.
    - **Tester d'autres modèles** : Essayer des modèles avancés comme LightGBM ou CatBoost.
    - **Déployer le modèle** : Intégrer le modèle dans un outil métier pour aider les conseillers.
    - **Surveiller la dérive** : Mettre en place un suivi de la performance du modèle dans le temps.
    
    ### Perspectives
    - Étendre l'analyse à d'autres produits bancaires ou à d'autres canaux de communication.
    - Utiliser des techniques d'explicabilité avancées pour mieux comprendre les décisions du modèle.
    """)

if selected == 'Outil  Prédictif':
    st.title("Outil Prédictif")
    st.write("Utilisez cet outil pour prédire la souscription d'un client à un dépôt à terme.")
    age = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
    balance = st.number_input("Solde du client", value=1000)
    duration = st.number_input("Durée de l'appel (en secondes)", value=100)
    campaign = st.number_input("Nombre de contacts pendant la campagne", value=1)
    previous = st.number_input("Nombre de contacts précédents", value=0)
    # Ajoutez d'autres champs selon vos besoins
    if st.button("Prédire"):
        # Dummy prediction (à remplacer par un vrai modèle)
        st.success("Résultat de la prédiction : Oui (exemple)")
