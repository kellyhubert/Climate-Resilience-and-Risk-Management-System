from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from .landslide_model import RwandaLandslideModel
from .flood_model import RwandaFloodModel
from .drought_model import RwandaDroughtModel

class RwandaMultiHazardModel:
    """Ensemble model that combines all three hazard models"""
    
    def __init__(self):
        self.landslide_model = RwandaLandslideModel('random_forest')
        self.flood_model = RwandaFloodModel('gradient_boosting')
        self.drought_model = RwandaDroughtModel('random_forest')
        self.models_trained = False
    
    def train_all_models(self, save_models=True, models_dir='models'):
        """Train all individual hazard models"""
        print("🚀 Training Multi-Hazard Ensemble for Rwanda")
        print("=" * 60)
        
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Train landslide model
        print("\n1️⃣ Training Landslide Model")
        landslide_results = self.landslide_model.train()
        
        # Train flood model
        print("\n2️⃣ Training Flood Model")
        flood_results = self.flood_model.train()
        
        # Train drought model
        print("\n3️⃣ Training Drought Model")
        drought_results = self.drought_model.train()
        
        self.models_trained = True
        
        # Save models
        if save_models:
            print("\n💾 Saving trained models...")
            self.landslide_model.save_model(models_dir / 'rwanda_landslide_model.pkl')
            self.flood_model.save_model(models_dir / 'rwanda_flood_model.pkl')
            self.drought_model.save_model(models_dir / 'rwanda_drought_model.pkl')
        
        # Generate comprehensive report
        self._generate_training_report(landslide_results, flood_results, drought_results)
        
        return {
            'landslide': landslide_results,
            'flood': flood_results,
            'drought': drought_results
        }
    
    def load_models(self, models_dir='models'):
        """Load pre-trained models"""
        models_dir = Path(models_dir)
        
        self.landslide_model.load_model(models_dir / 'rwanda_landslide_model.pkl')
        self.flood_model.load_model(models_dir / 'rwanda_flood_model.pkl')
        self.drought_model.load_model(models_dir / 'rwanda_drought_model.pkl')
        
        self.models_trained = True
        print("✅ All models loaded successfully")
    
    def predict_multi_hazard_risk(self, location_features):
        """Predict risk for all hazards at a given location"""
        if not self.models_trained:
            raise ValueError("Models must be trained or loaded first")
        
        results = {}
        
        # Landslide prediction
        if all(feature in location_features for feature in 
               ['elevation_m', 'slope_degrees', 'rainfall_24h_mm']):
            try:
                results['landslide'] = self.landslide_model.predict_risk_level(location_features)
            except Exception as e:
                results['landslide'] = {'error': str(e)}
        
        # Flood prediction
        if all(feature in location_features for feature in 
               ['elevation_m', 'rainfall_1h_mm', 'distance_to_river_m']):
            try:
                results['flood'] = self.flood_model.predict_risk_level(location_features)
            except Exception as e:
                results['flood'] = {'error': str(e)}
        
        # Drought prediction
        if all(feature in location_features for feature in 
               ['spi_3month', 'annual_rainfall_mm', 'ndvi']):
            try:
                results['drought'] = self.drought_model.predict_risk_level(location_features)
            except Exception as e:
                results['drought'] = {'error': str(e)}
        
        # Calculate overall risk
        risk_scores = []
        for hazard, result in results.items():
            if 'probability' in result:
                risk_scores.append(result['probability'])
        
        if risk_scores:
            overall_risk = max(risk_scores)  # Use maximum risk approach
            if overall_risk < 0.3:
                overall_level = "Low"
            elif overall_risk < 0.6:
                overall_level = "Medium"
            elif overall_risk < 0.8:
                overall_level = "High"
            else:
                overall_level = "Critical"
            
            results['overall'] = {
                'risk_score': overall_risk,
                'risk_level': overall_level,
                'dominant_hazard': max(results.keys(), 
                                     key=lambda x: results[x].get('probability', 0) 
                                     if 'probability' in results[x] else 0)
            }
        
        return results
    
    def _generate_training_report(self, landslide_results, flood_results, drought_results):
        """Generate comprehensive training report"""
        print("\n" + "="*60)
        print("📊 MULTI-HAZARD MODEL TRAINING REPORT")
        print("="*60)
        
        # Model performance summary
        print("Landslide Model:")
        print(f"  Algorithm: {self.landslide_model.algorithm}")
        print(f"  Accuracy: {self.landslide_model.training_history[-1]['test_accuracy']:.3f}")
        print("Flood Model:")
        print(f"  Algorithm: {self.flood_model.algorithm}")
        print(f"  Accuracy: {self.flood_model.training_history[-1]['test_accuracy']:.3f}")
        print("Drought Model:")
        print(f"  Algorithm: {self.drought_model.algorithm}")
        print(f"  Accuracy: {self.drought_model.training_history[-1]['test_accuracy']:.3f}")
    
    def prepare_data(self, data):
        """Prepare data for training/prediction"""
        # Handle categorical variables
        df = data.copy()
        
        # Encode categorical features
        categorical_features = ['soil_type', 'land_use', 'region']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
        
        # Select numerical features for model
        numerical_features = [
            'elevation_m', 'slope_degrees', 'aspect_degrees',
            'rainfall_24h_mm', 'rainfall_72h_mm', 'antecedent_rainfall_mm',
            'soil_moisture_fraction', 'soil_depth_m', 'ndvi',
            'distance_to_road_m', 'population_density_per_km2',
            'weathering_degree', 'fault_distance_m'
        ]
        
        # Add encoded categorical features
        encoded_features = [f'{feat}_encoded' for feat in categorical_features if feat in data.columns]
        self.features = numerical_features + encoded_features
        
        # Extract features that exist in the dataframe
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X
    
    def train(self, data=None, test_size=0.2, perform_cv=True):
        """Train the landslide model"""
        print(f"Training {self.model_name} using {self.algorithm}...")
        
        # Generate training data if not provided
        if data is None:
            data = self.generate_training_data(n_samples=5000)
        
        # Prepare features
        X = self.prepare_data(data)
        y = data['landslide_occurred'].values
        
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if len(X.columns) > 10:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(15, len(X.columns)))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Train model
        self.model.fit(X_train_selected, y_train)
        self.is_trained = True
        
        # Evaluate model
        train_score = self.model.score(X_train_selected, y_train)
        test_score = self.model.score(X_test_selected, y_test)
        
        # Cross-validation
        cv_scores = []
        if perform_cv:
            cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=5)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_selector:
                selected_features = [X.columns[i] for i in self.feature_selector.get_support(indices=True)]
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        # Store training results
        training_result = {
            'timestamp': datetime.now(),
            'algorithm': self.algorithm,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores) if cv_scores else None,
            'cv_std': np.std(cv_scores) if cv_scores else None,
            'n_features': X_train_selected.shape[1],
            'n_samples': X_train_selected.shape[0]
        }
        
        self.training_history.append(training_result)
        
        print(f"\nTraining Results:")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        if cv_scores:
            print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        return {
            'X_test': X_test_selected,
            'y_test': y_test,
            'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None
        }
    
    def predict_risk_level(self, features_dict):
        """Predict landslide risk level for a specific location"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Convert dict to DataFrame
        df = pd.DataFrame([features_dict])
        X = self.prepare_data(df)
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # Get predictions
        probability = self.model.predict_proba(X_selected)[0, 1]
        prediction = self.model.predict(X_selected)[0]
        
        # Convert to risk categories
        if probability < 0.2:
            risk_level = "Low"
        elif probability < 0.5:
            risk_level = "Medium"
        elif probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return {
            'probability': probability,
            'prediction': bool(prediction),
            'risk_level': risk_level,
            'confidence': max(probability, 1-probability)
        }