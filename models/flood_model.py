"""
Flood Risk Prediction Model for Rwanda

This module implements a machine learning model to predict flood risk in Rwanda
based on rainfall patterns, topography, drainage systems, and urbanization factors.
"""
from datetime import datetime
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, train_test_split
from base_model import BaseRiskModel
import numpy as np

class RwandaFloodModel(BaseRiskModel):
    """Advanced flood risk prediction model for Rwanda"""
    
    def __init__(self, model_algorithm='gradient_boosting'):
        """
        Initialize the flood risk model
        
        Args:
            model_algorithm (str): Algorithm to use ('gradient_boosting' or 'random_forest')
        """
        super().__init__("Rwanda_Flood_Model", "classification")
        self.algorithm = model_algorithm
        self.rwanda_params = self._setup_rwanda_parameters()
        self._initialize_model()
    
    def _setup_rwanda_parameters(self):
        """Rwanda-specific flood parameters based on local conditions"""
        return {
            'drainage_basins': {
                'Nyabarongo': {'flood_prone': 0.7, 'area_km2': 13500},
                'Akagera': {'flood_prone': 0.5, 'area_km2': 10300},
                'Mukungwa': {'flood_prone': 0.6, 'area_km2': 3000},
                'Nyabugogo': {'flood_prone': 0.8, 'area_km2': 850}  # Urban Kigali
            },
            'critical_rainfall_1h': 20,    # mm/hour flash flood threshold
            'critical_rainfall_6h': 50,    # mm/6h flood warning
            'critical_rainfall_24h': 100,  # mm/24h major flood
            'urban_drainage_capacity': 25, # mm/hour typical capacity
            'soil_infiltration_rate': 15   # mm/hour average infiltration
        }
    
    def _initialize_model(self):
        """Initialize the underlying ML model based on selected algorithm"""
        if self.algorithm == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=42
            )
        elif self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def generate_training_data(self, n_samples=4000):
        """
        Generate realistic flood training data for Rwanda
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            pd.DataFrame: Training data with features and labels
        """
        print(f"Generating {n_samples} training samples for flood model...")
        
        np.random.seed(42)
        data = []
        
        # Rwanda's major cities and flood-prone areas
        locations = {
            'Kigali_Urban': {'flood_risk': 0.6, 'elevation': 1500, 'drainage': 0.3},
            'Nyagatare_Plains': {'flood_risk': 0.7, 'elevation': 1350, 'drainage': 0.4},
            'Rusizi_Valley': {'flood_risk': 0.8, 'elevation': 1200, 'drainage': 0.2},
            'Huye_Town': {'flood_risk': 0.4, 'elevation': 1700, 'drainage': 0.6},
            'Rubavu_Lakeside': {'flood_risk': 0.5, 'elevation': 1460, 'drainage': 0.4}
        }
        
        for i in range(n_samples):
            # Select location
            location_name, location_params = np.random.choice(list(locations.items()))
            
            # Geographic features
            elevation = np.random.normal(location_params['elevation'], 100)
            elevation = np.clip(elevation, 900, 2500)
            
            slope = np.random.gamma(2, 5)  # Generally flatter areas flood more
            slope = np.clip(slope, 0, 30)
            
            # Distance to water bodies
            distance_to_river = np.random.gamma(2, 500)  # meters
            distance_to_lake = np.random.gamma(3, 2000)  # meters
            
            # Rainfall features (key flood trigger)
            rainfall_1h = np.random.gamma(1, 3)
            rainfall_6h = rainfall_1h + np.random.gamma(2, 8)
            rainfall_24h = rainfall_6h + np.random.gamma(2, 15)
            
            # Antecedent conditions
            antecedent_rainfall = np.random.gamma(2, 10)  # Previous 7 days
            soil_saturation = 0.2 + 0.6 * (antecedent_rainfall / 100)
            soil_saturation = np.clip(soil_saturation, 0.1, 0.95)
            
            # Urban features
            urbanization_degree = np.random.uniform(0, 1)
            if 'Urban' in location_name or 'Town' in location_name:
                urbanization_degree = np.random.uniform(0.6, 1.0)
            
            impervious_surface_pct = urbanization_degree * 80  # % of area that's paved
            
            # Drainage characteristics
            drainage_density = location_params['drainage'] + np.random.normal(0, 0.2)
            drainage_density = np.clip(drainage_density, 0.1, 1.0)
            
            drainage_condition = np.random.uniform(0.2, 1.0)  # 1 = excellent, 0 = blocked
            
            # Vegetation and land use
            vegetation_cover = np.random.uniform(0.1, 0.9)
            land_use = np.random.choice(['urban', 'agricultural', 'forest', 'wetland', 'grassland'])
            
            # Population and infrastructure
            population_density = np.random.gamma(2, 200)
            building_density = population_density * 0.1
            
            # Calculate flood probability
            prob = self._calculate_flood_probability({
                'elevation': elevation, 'slope': slope,
                'distance_to_river': distance_to_river, 'distance_to_lake': distance_to_lake,
                'rainfall_1h': rainfall_1h, 'rainfall_6h': rainfall_6h, 'rainfall_24h': rainfall_24h,
                'antecedent_rainfall': antecedent_rainfall, 'soil_saturation': soil_saturation,
                'urbanization_degree': urbanization_degree, 'impervious_surface_pct': impervious_surface_pct,
                'drainage_density': drainage_density, 'drainage_condition': drainage_condition,
                'vegetation_cover': vegetation_cover, 'location_flood_risk': location_params['flood_risk']
            })
            
            # Generate binary label
            flood_occurred = np.random.random() < prob
            
            # Create record
            record = {
                'elevation_m': elevation,
                'slope_degrees': slope,
                'distance_to_river_m': distance_to_river,
                'distance_to_lake_m': distance_to_lake,
                'rainfall_1h_mm': rainfall_1h,
                'rainfall_6h_mm': rainfall_6h,
                'rainfall_24h_mm': rainfall_24h,
                'antecedent_rainfall_mm': antecedent_rainfall,
                'soil_saturation_fraction': soil_saturation,
                'urbanization_degree': urbanization_degree,
                'impervious_surface_pct': impervious_surface_pct,
                'drainage_density': drainage_density,
                'drainage_condition': drainage_condition,
                'vegetation_cover_fraction': vegetation_cover,
                'land_use': land_use,
                'population_density_per_km2': population_density,
                'building_density_per_km2': building_density,
                'location': location_name,
                'flood_occurred': int(flood_occurred)
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        print(f"✓ Generated dataset: {len(df)} samples")
        print(f"✓ Flood events: {df['flood_occurred'].sum()} ({df['flood_occurred'].mean()*100:.1f}%)")
        
        return df
    
    def _calculate_flood_probability(self, features):
        """Calculate flood probability based on Rwanda-specific factors"""
        prob = 0.0
        
        # Rainfall intensity factors (primary trigger)
        rainfall_1h_factor = min(1.0, features['rainfall_1h'] / 30)
        rainfall_6h_factor = min(1.0, features['rainfall_6h'] / 80)
        rainfall_24h_factor = min(1.0, features['rainfall_24h'] / 150)
        
        # Use the most critical rainfall metric
        max_rainfall_factor = max(rainfall_1h_factor, rainfall_6h_factor * 0.8, rainfall_24h_factor * 0.6)
        prob += max_rainfall_factor * 0.30
        
        # Topographic factors
        elevation_factor = max(0, (1800 - features['elevation']) / 800)  # Lower = more prone
        slope_factor = max(0, (20 - features['slope']) / 20)  # Flatter = more prone
        prob += elevation_factor * 0.15
        prob += slope_factor * 0.10
        
        # Proximity to water bodies
        river_factor = 1 / (1 + features['distance_to_river'] / 200)
        lake_factor = 1 / (1 + features['distance_to_lake'] / 1000)
        prob += max(river_factor, lake_factor * 0.5) * 0.10
        
        # Urban drainage factors
        urbanization_factor = features['urbanization_degree']
        impervious_factor = features['impervious_surface_pct'] / 100
        drainage_factor = 1 - features['drainage_condition']  # Poor drainage = higher risk
        
        prob += urbanization_factor * 0.10
        prob += impervious_factor * 0.08
        prob += drainage_factor * 0.07
        
        # Antecedent conditions
        saturation_factor = max(0, (features['soil_saturation'] - 0.5) / 0.4)
        prob += saturation_factor * 0.05
        
        # Vegetation factor (less vegetation = higher runoff)
        vegetation_factor = max(0, (0.7 - features['vegetation_cover']) / 0.7)
        prob += vegetation_factor * 0.05
        
        # Location base risk
        prob += features['location_flood_risk'] * 0.05
        
        return np.clip(prob, 0, 1)
    
    def prepare_data(self, data):
        """
        Prepare data for training/prediction
        
        Args:
            data (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Prepared feature matrix
        """
        df = data.copy()
        
        # Encode categorical features
        categorical_features = ['land_use', 'location']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
        
        # Select numerical features
        numerical_features = [
            'elevation_m', 'slope_degrees', 'distance_to_river_m', 'distance_to_lake_m',
            'rainfall_1h_mm', 'rainfall_6h_mm', 'rainfall_24h_mm', 'antecedent_rainfall_mm',
            'soil_saturation_fraction', 'urbanization_degree', 'impervious_surface_pct',
            'drainage_density', 'drainage_condition', 'vegetation_cover_fraction',
            'population_density_per_km2', 'building_density_per_km2'
        ]
        
        # Add encoded categorical features
        encoded_features = [f'{feat}_encoded' for feat in categorical_features if feat in data.columns]
        self.features = numerical_features + encoded_features
        
        # Extract available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X
    
    def train(self, data=None, test_size=0.2, perform_cv=True):
        """
        Train the flood model
        
        Args:
            data (pd.DataFrame, optional): Training data. If None, generates synthetic data
            test_size (float): Proportion of data for testing
            perform_cv (bool): Whether to perform cross-validation
            
        Returns:
            dict: Training results including test data and feature importance
        """
        print(f"Training {self.model_name} using {self.algorithm}...")
        
        # Generate training data if not provided
        if data is None:
            data = self.generate_training_data(n_samples=4000)
        
        # Prepare features
        X = self.prepare_data(data)
        y = data['flood_occurred'].values
        
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
        if len(X.columns) > 12:
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(12, len(X.columns)))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Train model
        self.model.fit(X_train_selected, y_train)
        self.is_trained = True
        
        # Evaluate
        train_score = self.model.score(X_train_selected, y_train)
        test_score = self.model.score(X_test_selected, y_test)
        
        # Cross-validation
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
        
        print(f"\n✓ Training Results:")
        print(f"  Training Accuracy: {train_score:.3f}")
        print(f"  Test Accuracy: {test_score:.3f}")
        if cv_scores:
            print(f"  CV Score: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        return {
            'X_test': X_test_selected,
            'y_test': y_test,
            'feature_importance': feature_importance
        }
    
    def predict_risk_level(self, features_dict):
        """
        Predict flood risk level for a specific location
        
        Args:
            features_dict (dict): Dictionary containing feature values
            
        Returns:
            dict: Prediction results with probability, risk level, and confidence
        """
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
        if probability < 0.25:
            risk_level = "Low"
        elif probability < 0.5:
            risk_level = "Medium"
        elif probability < 0.8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        
        return {
            'probability': float(probability),
            'prediction': bool(prediction),
            'risk_level': risk_level,
            'confidence': float(max(probability, 1-probability))
        }