"""
Drought Risk Prediction Model for Rwanda

This module implements a machine learning model to predict drought risk in Rwanda
based on rainfall patterns, temperature, vegetation indices, and socio-economic factors.
"""
from datetime import datetime
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from .base_model import BaseRiskModel

class RwandaDroughtModel(BaseRiskModel):
    """Advanced drought risk prediction model for Rwanda"""
    
    def __init__(self, model_algorithm='random_forest'):
        super().__init__("Rwanda_Drought_Model", "classification")
        self.algorithm = model_algorithm
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.feature_selector = None
        
        # Setup Rwanda-specific parameters
        self.rwanda_params = self._setup_rwanda_parameters()
        
        # Initialize the ML model
        self._initialize_model()
        
        # Store model parameters for saving/loading
        self.model_params = {
            'algorithm': self.algorithm,
            'rwanda_specific_params': self.rwanda_params,
            'model_type': 'drought_risk',
            'version': '1.0'
        }
    
    def _setup_rwanda_parameters(self):
        """Rwanda-specific drought parameters"""
        return {
            'climate_zones': {
                'Eastern_Dry': {'annual_rainfall': 800, 'drought_freq': 0.3},
                'Central_Moderate': {'annual_rainfall': 1200, 'drought_freq': 0.15},
                'Western_Wet': {'annual_rainfall': 1600, 'drought_freq': 0.1},
                'Northern_Variable': {'annual_rainfall': 1100, 'drought_freq': 0.2}
            },
            'spi_thresholds': {
                'mild_drought': -1.0,
                'moderate_drought': -1.5,
                'severe_drought': -2.0
            },
            'growing_seasons': {
                'Season_A': {'start_month': 9, 'end_month': 1},    # Sep-Jan
                'Season_B': {'start_month': 2, 'end_month': 6},    # Feb-Jun
                'Season_C': {'start_month': 6, 'end_month': 9}     # Jun-Sep (dry season)
            }
        }
    
    def _initialize_model(self):
        """Initialize the underlying ML model"""
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=250,
                max_depth=10,
                min_samples_split=15,
                random_state=42,
                class_weight='balanced_subsample'
            )
        elif self.algorithm == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.08,
                max_depth=5,
                random_state=42
            )
    
    def generate_training_data(self, n_samples=6000):
        """Generate realistic drought training data for Rwanda"""
        print(f"Generating {n_samples} training samples for drought model...")
        
        np.random.seed(42)
        data = []
        
        climate_zones = self.rwanda_params['climate_zones']
        
        for i in range(n_samples):
            # Select climate zone - FIX: Use random.choice instead of np.random.choice
            zone_name, zone_params = random.choice(list(climate_zones.items()))
            
            # Temporal features
            month = np.random.randint(1, 13)
            year_day = np.random.randint(1, 366)
            
            # Long-term precipitation patterns
            annual_rainfall_normal = zone_params['annual_rainfall']
            
            # Generate rainfall time series (simplified)
            current_month_rainfall = np.random.gamma(2, annual_rainfall_normal / 24)  # Monthly
            previous_3month_rainfall = current_month_rainfall + np.random.gamma(3, annual_rainfall_normal / 8)
            previous_6month_rainfall = previous_3month_rainfall + np.random.gamma(4, annual_rainfall_normal / 6)
            annual_rainfall = previous_6month_rainfall + np.random.gamma(6, annual_rainfall_normal / 4)
            
            # Calculate SPI (Standardized Precipitation Index) approximation
            rainfall_anomaly = (annual_rainfall - annual_rainfall_normal) / (annual_rainfall_normal * 0.3)
            spi_3month = np.clip(rainfall_anomaly + np.random.normal(0, 0.5), -3, 3)
            spi_6month = np.clip(rainfall_anomaly * 0.8 + np.random.normal(0, 0.4), -3, 3)
            spi_12month = np.clip(rainfall_anomaly * 0.7 + np.random.normal(0, 0.3), -3, 3)
            
            # Temperature patterns
            base_temp = 20 + 3 * np.sin((month - 1) * np.pi / 6)  # Seasonal variation
            temperature_anomaly = np.random.normal(0, 2)
            avg_temperature = base_temp + temperature_anomaly
            max_temperature = avg_temperature + np.random.uniform(5, 12)
            
            # Vegetation and agricultural indicators
            base_ndvi = 0.5 + 0.2 * np.sin((month - 3) * np.pi / 6)  # Seasonal vegetation
            rainfall_effect = max(-0.3, min(0.3, (annual_rainfall - annual_rainfall_normal) / annual_rainfall_normal))
            ndvi = base_ndvi + rainfall_effect * 0.4 + np.random.normal(0, 0.05)
            ndvi = np.clip(ndvi, 0.1, 0.9)
            
            # Soil moisture estimation
            base_soil_moisture = 0.4
            precip_effect = (current_month_rainfall / 100) * 0.3
            temp_effect = -((avg_temperature - 22) / 10) * 0.1
            soil_moisture = base_soil_moisture + precip_effect + temp_effect + np.random.normal(0, 0.1)
            soil_moisture = np.clip(soil_moisture, 0.05, 0.8)
            
            # Hydrological indicators
            river_flow_index = 0.5 + (previous_6month_rainfall - annual_rainfall_normal/2) / (annual_rainfall_normal/2)
            river_flow_index = np.clip(river_flow_index, 0.1, 2.0)
            
            # Agricultural stress indicators
            crop_stress_days = max(0, int(30 * (1 - min(1, annual_rainfall / annual_rainfall_normal))))
            consecutive_dry_days = max(0, int(45 * (1 - current_month_rainfall / (annual_rainfall_normal/12))))
            
            # Economic and social factors
            irrigation_coverage = np.random.uniform(0.1, 0.6)  # % of agricultural area irrigated
            crop_diversity = np.random.uniform(0.3, 0.9)  # Agricultural diversity index
            water_storage_capacity = np.random.uniform(0.2, 0.8)  # Community water storage
            
            # Calculate drought probability
            prob = self._calculate_drought_probability({
                'spi_3month': spi_3month, 'spi_6month': spi_6month, 'spi_12month': spi_12month,
                'current_month_rainfall': current_month_rainfall, 'annual_rainfall': annual_rainfall,
                'annual_rainfall_normal': annual_rainfall_normal, 'avg_temperature': avg_temperature,
                'max_temperature': max_temperature, 'ndvi': ndvi, 'soil_moisture': soil_moisture,
                'river_flow_index': river_flow_index, 'consecutive_dry_days': consecutive_dry_days,
                'irrigation_coverage': irrigation_coverage, 'crop_diversity': crop_diversity,
                'zone_drought_freq': zone_params['drought_freq'], 'month': month
            })
            
            # Generate binary label
            drought_occurred = np.random.random() < prob
            
            # Create record
            record = {
                'month': month,
                'spi_3month': spi_3month,
                'spi_6month': spi_6month,
                'spi_12month': spi_12month,
                'current_month_rainfall_mm': current_month_rainfall,
                'previous_3month_rainfall_mm': previous_3month_rainfall,
                'previous_6month_rainfall_mm': previous_6month_rainfall,
                'annual_rainfall_mm': annual_rainfall,
                'annual_rainfall_normal_mm': annual_rainfall_normal,
                'rainfall_deficit_pct': ((annual_rainfall_normal - annual_rainfall) / annual_rainfall_normal) * 100,
                'avg_temperature_c': avg_temperature,
                'max_temperature_c': max_temperature,
                'temperature_anomaly_c': temperature_anomaly,
                'ndvi': ndvi,
                'soil_moisture_fraction': soil_moisture,
                'river_flow_index': river_flow_index,
                'consecutive_dry_days': consecutive_dry_days,
                'crop_stress_days': crop_stress_days,
                'irrigation_coverage_pct': irrigation_coverage * 100,
                'crop_diversity_index': crop_diversity,
                'water_storage_capacity': water_storage_capacity,
                'climate_zone': zone_name,
                'drought_occurred': int(drought_occurred)
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        print(f"Generated dataset: {len(df)} samples")
        print(f"Drought events: {df['drought_occurred'].sum()} ({df['drought_occurred'].mean()*100:.1f}%)")
        
        return df
    
    def _calculate_drought_probability(self, features):
        """Calculate drought probability based on Rwanda-specific factors"""
        prob = 0.0
        
        # SPI indicators (most important for meteorological drought)
        spi_3 = float(features['spi_3month'])
        spi_6 = float(features['spi_6month'])
        spi_12 = float(features['spi_12month'])
        
        spi_3_factor = float(np.clip((-spi_3 - 0.5) / 2.5, 0, 1)) if spi_3 < 0 else 0
        spi_6_factor = float(np.clip((-spi_6 - 0.5) / 2.5, 0, 1)) if spi_6 < 0 else 0
        spi_12_factor = float(np.clip((-spi_12 - 0.5) / 2.5, 0, 1)) if spi_12 < 0 else 0
        
        prob += max(spi_3_factor * 0.25, spi_6_factor * 0.20, spi_12_factor * 0.15)
        
        # Rainfall deficit
        annual_rainfall = float(features['annual_rainfall'])
        annual_rainfall_normal = float(features['annual_rainfall_normal'])
        rainfall_deficit = float(np.clip((annual_rainfall_normal - annual_rainfall) / annual_rainfall_normal, 0, 1))
        prob += rainfall_deficit * 0.20
        
        # Temperature stress
        avg_temp = float(features['avg_temperature'])
        temp_stress = float(np.clip((avg_temp - 24) / 10, 0, 1))
        prob += temp_stress * 0.10
        
        # Vegetation stress (NDVI)
        ndvi = float(features['ndvi'])
        vegetation_stress = float(np.clip((0.4 - ndvi) / 0.4, 0, 1))
        prob += vegetation_stress * 0.15
        
        # Soil moisture deficit
        soil_moisture = float(features['soil_moisture'])
        soil_stress = float(np.clip((0.3 - soil_moisture) / 0.3, 0, 1))
        prob += soil_stress * 0.10
        
        # Hydrological stress
        river_flow = float(features['river_flow_index'])
        hydro_stress = float(np.clip((0.7 - river_flow) / 0.7, 0, 1))
        prob += hydro_stress * 0.08
        
        # Consecutive dry days
        dry_days = float(features['consecutive_dry_days'])
        dry_days_stress = float(np.clip(dry_days / 60, 0, 1))
        prob += dry_days_stress * 0.07
        
        # Adaptive capacity (reduces drought impact)
        irrigation = float(features['irrigation_coverage'])
        crop_div = float(features['crop_diversity'])
        adaptive_capacity = (irrigation + crop_div) / 2
        prob *= (1 - adaptive_capacity * 0.3)
        
        # Seasonal factor
        month = int(features['month'])
        dry_season_months = [6, 7, 8, 9]  # Season C
        if month in dry_season_months:
            prob *= 1.3
        
        # Climate zone base risk
        prob += float(features['zone_drought_freq']) * 0.05
        
        return float(np.clip(prob, 0, 1))
    
    def prepare_data(self, data):
        """Prepare drought data for training/prediction"""
        df = data.copy()
        
        # Encode categorical features
        if 'climate_zone' in df.columns:
            le = LabelEncoder()
            df['climate_zone_encoded'] = le.fit_transform(df['climate_zone'].astype(str))
        
        # Select features
        numerical_features = [
            'month', 'spi_3month', 'spi_6month', 'spi_12month',
            'current_month_rainfall_mm', 'previous_3month_rainfall_mm',
            'previous_6month_rainfall_mm', 'annual_rainfall_mm',
            'rainfall_deficit_pct', 'avg_temperature_c', 'max_temperature_c',
            'temperature_anomaly_c', 'ndvi', 'soil_moisture_fraction',
            'river_flow_index', 'consecutive_dry_days', 'crop_stress_days',
            'irrigation_coverage_pct', 'crop_diversity_index', 'water_storage_capacity'
        ]
        
        encoded_features = ['climate_zone_encoded'] if 'climate_zone' in data.columns else []
        self.features = numerical_features + encoded_features
        
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        X = X.fillna(X.mean())
        
        return X
    
    def train(self, data=None, test_size=0.2, perform_cv=True):
        """Train the drought model"""
        print(f"Training {self.model_name} using {self.algorithm}...")
        
        if data is None:
            data = self.generate_training_data(n_samples=6000)
        
        X = self.prepare_data(data)
        y = data['drought_occurred'].values
        
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection (keep more features for drought as it's complex)
        if len(X.columns) > 15:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(15, len(X.columns)))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Train
        self.model.fit(X_train_selected, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_selected, y_train)
        test_score = self.model.score(X_test_selected, y_test)
        
        cv_scores = []
        if perform_cv:
            cv_scores = cross_val_score(self.model, X_train_selected, y_train, cv=5)
        
        # Feature importance
        feature_importance = None
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
        
        # Store results - FIX: Use len() for array check
        training_result = {
            'timestamp': datetime.now(),
            'algorithm': self.algorithm,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores) if len(cv_scores) > 0 else None,
            'cv_std': np.std(cv_scores) if len(cv_scores) > 0 else None
        }
        
        self.training_history.append(training_result)
        
        print(f"\nTraining Results:")
        print(f"Training Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        if len(cv_scores) > 0:
            print(f"CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        return {
            'X_test': X_test_selected,
            'y_test': y_test,
            'feature_importance': feature_importance
        }
    
    def predict_risk_level(self, features_dict):
        """Predict drought risk level for a specific location/time"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        df = pd.DataFrame([features_dict])
        X = self.prepare_data(df)
        
        X_scaled = self.scaler.transform(X)
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        probability = self.model.predict_proba(X_selected)[0, 1]
        prediction = self.model.predict(X_selected)[0]
        
        if probability < 0.2:
            risk_level = "Low"
        elif probability < 0.4:
            risk_level = "Medium"
        elif probability < 0.7:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return {
            'probability': float(probability),
            'prediction': bool(prediction),
            'risk_level': risk_level,
            'confidence': float(max(probability, 1-probability))
        }