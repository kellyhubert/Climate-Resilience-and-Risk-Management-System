"""
Base model class for risk assessment models."""

import pickle
import joblib
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class BaseRiskModel:
    """Base class for all Risk assessment Models."""

    def __init__(self, model_name, model_type='classification'):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()  # Initialize scaler
        self.feature_selector = None
        self.features = []
        self.model_params = {}
        self.is_trained = False
        self.training_history = []

    def prepare_data(self, data):
        """Prepare data for training or prediction."""
        raise NotImplementedError("Must implement prepare_data method")
    
    def train(self, X, y):  # Fixed typo: trin -> train
        """Train the Risk model"""
        raise NotImplementedError("Must implement train method")
    
    def predict(self, X):
        """Make prediction using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)  # Fixed typo: Predict_proba -> predict_proba
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        probabilities = self.model.predict_proba(X_test)  # Fixed typo: Predict_proba -> predict_proba

        evaluation = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),  # Fixed typo: confision_matrix -> confusion_matrix
            'model_name': self.model_name,
            'test_samples': len(X_test),
            'evaluation_date': datetime.now().isoformat()
        }

        # Add AUC score if probabilities available
        if probabilities is not None and len(np.unique(y_test)) == 2:
            evaluation['auc_score'] = roc_auc_score(y_test, probabilities[:, 1]) 

        return evaluation
       
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'features': self.features,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'model_params': self.model_params,
            'save_date': datetime.now().isoformat(),  # Fixed typo: save_model -> save_date
        }

        filepath = Path(filepath)
        if filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        elif filepath.suffix == '.joblib':
            joblib.dump(model_data, filepath)
        else:
            # Default to pickle if no extension or unknown extension
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):  # Fixed typo: lead_model -> load_model
        """Load trained model from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        elif filepath.suffix == '.joblib':
            model_data = joblib.load(filepath)
        else:
            # Try pickle as default
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data.get('scaler', StandardScaler())
        self.feature_selector = model_data.get('feature_selector', None)
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.features = model_data.get('features', [])
        self.is_trained = model_data.get('is_trained', True)
        self.training_history = model_data.get('training_history', [])
        self.model_params = model_data.get('model_params', {})

        print(f"Model loaded from: {filepath}")