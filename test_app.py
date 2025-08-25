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

st.title("Test App - Bank Marketing Prediction")
st.write("Testing basic functionality...")

# Test data loading
try:
    df = pd.read_csv('bank.csv')
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")

# Test basic operations
try:
    st.write("✅ All imports successful!")
    st.write(f"✅ Scikit-learn version: {sklearn.__version__}")
    st.write(f"✅ XGBoost version: {xgb.__version__}")
    st.write("✅ Streamlit components loaded!")
except Exception as e:
    st.error(f"❌ Error in basic operations: {e}")

st.write("🎉 Test completed successfully!")
