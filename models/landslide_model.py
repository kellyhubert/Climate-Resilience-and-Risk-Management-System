from datetime import datetime
import random
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from .base_model import BaseRiskModel

class RwandaLandslideModel(BaseRiskModel):
    """Advanced Landslide Risk prediction Model for Rwanda."""
    
    def __init__(self, model_algorithm='RandomForest'):
        super().__init__("Rwanda_Landslide_Model", "classification")
        self.algorithm = model_algorithm

        self.scaler = StandardScaler()
        self.feature_selector = None

        self.rwanda_params = self._setup_rwanda_parameters()
        self._initialize_model()

        # Add model_params for save_model() compatibility
        self.model_params = {
            'algorithm': self.algorithm,
            'n_estimators': 200 if self.algorithm == 'RandomForest' else 150,
            'rwanda_specific_params': self.rwanda_params,
            'model_type': 'landslide_risk',
            'version': '1.0'
         }

    def _setup_rwanda_parameters(self):
        """Rwanda specific landslide parameters."""
        return {
            'elevation_range': (900, 4507),
            'critical_slope': 25,
            'rainfall_threshold': 50,
            'soil_saturation_threshold': 0.8,
            'vegetation_threshold': 0.3,
            'geology_weight': {
                'volcanic': 0.6,
                'sedimentary': 0.8,
                'weathered': 1.0,
                'alluvial': 0.7
            }
        }

    def _initialize_model(self):
        """Initialize the underlying ML model"""
        if self.algorithm == 'RandomForest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )
        elif self.algorithm == 'GradientBoosting':
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        elif self.algorithm == 'LogisticRegression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
    
    def generate_training_data(self, n_samples=5000):
        """Generate realistic training data for Rwanda landslides"""
        print(f"Generating {n_samples} training samples for landslide model...")

        np.random.seed(42)
        data = []

        # Rwanda's geographic regions with different landslide susceptibilities
        regions = {
            'Northern_Mountains': {'base_risk': 0.7, 'elevation_mean': 2200, 'slope_mean': 30},
            'Central_Plateau': {'base_risk': 0.4, 'elevation_mean': 1600, 'slope_mean': 20},
            'Eastern_Hills': {'base_risk': 0.5, 'elevation_mean': 1400, 'slope_mean': 25},
            'Western_Ridges': {'base_risk': 0.6, 'elevation_mean': 1800, 'slope_mean': 28},
            'Southern_Valleys': {'base_risk': 0.3, 'elevation_mean': 1300, 'slope_mean': 15}
        }

        for i in range(n_samples):
            # Select random region
            region_name, region_params = random.choice(list(regions.items()))

            # Terrain features
            elevation = np.random.normal(region_params['elevation_mean'], 200)
            elevation = np.clip(elevation, 900, 4500)

            # FIX: Use slope_mean instead of elevation_mean
            slope = np.random.normal(region_params['slope_mean'], 8)
            slope = np.clip(slope, 0, 60)

            aspect = np.random.uniform(0, 360)

            # Generate climate features - Rainfall pattern (more in wet season)
            season_factor = np.sin((i % 365) * 2 * np.pi / 365) * 0.3 + 1  # Seasonal variation
            daily_rainfall = np.random.gamma(2, 3) * season_factor
            
            # Multiple rainfall timeframes
            rainfall_1h = daily_rainfall * 0.2 + np.random.gamma(1, 2)
            rainfall_6h = daily_rainfall * 0.6 + np.random.gamma(1, 5)
            rainfall_24h = daily_rainfall + np.random.gamma(1, 10)  # Extreme event
            rainfall_72h = rainfall_24h + np.random.gamma(1, 15)
            antecedent_rainfall = np.random.gamma(3, 8)  # Previous week rainfall

            # Soil and vegetation features
            soil_depth = np.random.gamma(2, 1.5)  # meters
            soil_type = random.choice(['volcanic', 'sedimentary', 'weathered', 'alluvial'])
            soil_moisture = 0.3 + 0.4 * (rainfall_24h / 100) + np.random.normal(0, 0.1)
            soil_moisture = np.clip(soil_moisture, 0.1, 0.95)
            soil_saturation = soil_moisture  # Use same value for saturation

            # Vegetation (NDVI)
            base_ndvi = 0.4 + (elevation - 1000) / 3000 * 0.3  # Higher elevation = more vegetation
            seasonal_ndvi = 0.2 * np.sin((i % 365) * 2 * np.pi / 365 + np.pi/2)  # Peak in wet season
            ndvi = base_ndvi + seasonal_ndvi + np.random.normal(0, 0.05)
            ndvi = np.clip(ndvi, 0.1, 0.9)
            vegetation_cover = ndvi  # Use NDVI as vegetation cover

            # Human factors
            distance_to_road = np.random.gamma(2, 200)  # meters
            land_use = np.random.choice(['forest', 'agriculture', 'urban', 'grassland'])
            population_density = np.random.gamma(1, 100)  # people per km2

            # Water body distances
            distance_to_river = np.random.gamma(2, 500)  # meters
            distance_to_lake = np.random.gamma(3, 1000)  # meters
            
            # Proximity factors (normalized 0-1, closer = higher value)
            river_proximity = 1 - min(1.0, distance_to_river / 2000)
            lake_proximity = 1 - min(1.0, distance_to_lake / 3000)

            # Urbanization features
            if land_use == 'urban':
                urbanization_degree = 0.7 + np.random.uniform(-0.1, 0.2)
                impervious_surface_pct = 60 + np.random.uniform(0, 30)
                building_density = population_density * 0.01
            elif land_use == 'agriculture':
                urbanization_degree = 0.2 + np.random.uniform(-0.1, 0.1)
                impervious_surface_pct = 10 + np.random.uniform(0, 20)
                building_density = population_density * 0.003
            else:
                urbanization_degree = 0.05 + np.random.uniform(0, 0.1)
                impervious_surface_pct = 5 + np.random.uniform(0, 10)
                building_density = population_density * 0.001
            
            urbanization_degree = np.clip(urbanization_degree, 0, 1)
            impervious_surface_pct = np.clip(impervious_surface_pct, 0, 100)
            building_density = np.clip(building_density, 0, 50)

            # Drainage features
            drainage_density = (rainfall_24h / 50) + (slope / 100) + np.random.uniform(0, 2)
            drainage_condition = np.random.uniform(1, 5)  # 1=poor, 5=excellent

            # Geological features
            weathering_degree = np.random.uniform(1, 5)  # 1=fresh, 5=highly weathered
            fault_distance = np.random.gamma(2, 1000)  # meters to nearest fault
            
            # Location-based flood risk (for multi-hazard context)
            location_flood_risk = region_params['base_risk'] * 0.5 + np.random.uniform(0, 0.3)
            
            # Calculate landslide probability using Rwanda-specific factors
            prob = self._calculate_landslide_probability({
                'elevation': elevation,
                'slope': slope,
                'aspect': aspect,
                'rainfall_24h': rainfall_24h,
                'rainfall_72h': rainfall_72h,
                'antecedent_rainfall': antecedent_rainfall,
                'soil_moisture': soil_moisture,
                'soil_saturation': soil_saturation,
                'soil_depth': soil_depth,
                'soil_type': soil_type,
                'ndvi': ndvi,
                'vegetation_cover': vegetation_cover,
                'distance_to_road': distance_to_road,
                'distance_to_river': distance_to_river,
                'distance_to_lake': distance_to_lake,
                'river_proximity': river_proximity,
                'lake_proximity': lake_proximity,
                'land_use': land_use,
                'urbanization_degree': urbanization_degree,
                'impervious_surface_pct': impervious_surface_pct,
                'drainage_condition': drainage_condition,
                'weathering_degree': weathering_degree,
                'fault_distance': fault_distance,
                'location_flood_risk': location_flood_risk,
                'region_base_risk': region_params['base_risk']
            })
            
            # Generate binary label with some randomness
            landslide_occurred = np.random.random() < prob
            
            # Create record with ALL required features
            record = {
                'elevation_m': elevation,
                'slope_degrees': slope,
                'aspect_degrees': aspect,
                'distance_to_river_m': distance_to_river,
                'distance_to_lake_m': distance_to_lake,
                'rainfall_1h_mm': rainfall_1h,
                'rainfall_6h_mm': rainfall_6h,
                'rainfall_24h_mm': rainfall_24h,
                'rainfall_72h_mm': rainfall_72h,
                'antecedent_rainfall_mm': antecedent_rainfall,
                'soil_moisture_fraction': soil_moisture,
                'soil_saturation_fraction': soil_saturation,
                'soil_depth_m': soil_depth,
                'soil_type': soil_type,
                'ndvi': ndvi,
                'vegetation_cover_fraction': vegetation_cover,
                'urbanization_degree': urbanization_degree,
                'impervious_surface_pct': impervious_surface_pct,
                'drainage_density': drainage_density,
                'drainage_condition': drainage_condition,
                'distance_to_road_m': distance_to_road,
                'land_use': land_use,
                'population_density_per_km2': population_density,
                'building_density_per_km2': building_density,
                'weathering_degree': weathering_degree,
                'fault_distance_m': fault_distance,
                'region': region_name,
                'location': region_name,  # For categorical encoding
                'landslide_occurred': int(landslide_occurred)
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        print(f"Generated dataset: {len(df)} samples")
        print(f"Landslide events: {df['landslide_occurred'].sum()} ({df['landslide_occurred'].mean()*100:.1f}%)")
        
        return df
    
    def _calculate_landslide_probability(self, features):
        """Calculate landslide probability based on Rwanda-specific factors"""
        prob = 0.0
        
        def get_value(key, default=0):
            val = features.get(key, default)
            # Convert to scalar if it's an array
            if isinstance(val, (np.ndarray, pd.Series)):
                return float(val.item() if val.size == 1 else val.mean())
            return float(val)

        # Slope factor (most critical in Rwanda's hilly terrain)
        slope = get_value('slope')
        slope_factor = min(1.0, max(0, (features['slope'] - 10) / 40))
        prob += slope_factor * 0.25
        
        # Rainfall factors (trigger events)
        rainfall_24h_factor = min(1.0, features['rainfall_24h'] / 80)
        rainfall_72h_factor = min(1.0, features['rainfall_72h'] / 150)
        antecedent_factor = min(1.0, features['antecedent_rainfall'] / 100)
        prob += (rainfall_24h_factor * 0.15 + rainfall_72h_factor * 0.10 + antecedent_factor * 0.08)
        
        # Water body proximity factors
        river_factor = features.get('river_proximity', 0)  # Higher = closer
        lake_factor = features.get('lake_proximity', 0)
        prob += max(river_factor, lake_factor * 0.5) * 0.05
        
        # Urban drainage factors
        urbanization_factor = get_value('urbanization_degree')
        impervious_factor = get_value('impervious_surface_pct') / 100
        drainage_cond = get_value('drainage_condition')
        drainage_factor = (5 - drainage_cond) / 4   # Poor drainage = higher risk
        
        prob += urbanization_factor * 0.08
        prob += impervious_factor * 0.06
        prob += drainage_factor * 0.05
        
        # Soil saturation
        soil_sat = get_value('soil_saturation')
        saturation_factor = max(0, (soil_sat - 0.5) / 0.4)
        prob += saturation_factor * 0.10
        
        # Vegetation factor (less vegetation = higher runoff)
        veg_cover = get_value('vegetation_cover')
        vegetation_factor = max(0, (0.7 - veg_cover) / 0.7)
        prob += vegetation_factor * 0.08
        
        # Location base risk
        location_risk = get_value('location_flood_risk', 0.3)
        prob += location_risk * 0.05
        
        return np.clip(prob, 0, 1)
    
    def prepare_data(self, data):
        """Prepare landslide data for training/prediction"""
        df = data.copy()
        
        # Encode categorical features
        categorical_features = ['land_use', 'location']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
        
        # Select features
        numerical_features = [
            'elevation_m', 'slope_degrees', 'distance_to_river_m', 'distance_to_lake_m',
            'rainfall_1h_mm', 'rainfall_6h_mm', 'rainfall_24h_mm', 'antecedent_rainfall_mm',
            'soil_saturation_fraction', 'urbanization_degree', 'impervious_surface_pct',
            'drainage_density', 'drainage_condition', 'vegetation_cover_fraction',
            'population_density_per_km2', 'building_density_per_km2'
        ]
        
        encoded_features = [f'{feat}_encoded' for feat in categorical_features if feat in data.columns]
        self.features = numerical_features + encoded_features
        
        # Extract available features
        available_features = [f for f in self.features if f in df.columns]
        X = df[available_features]
        X = X.fillna(X.mean())
        
        return X
    
    def train(self, data=None, test_size=0.2, perform_cv=True):
        """Train the landslide model"""
        print(f"Training {self.model_name} using {self.algorithm}...")
        
        if data is None:
            data = self.generate_training_data(n_samples=4000)
        
        X = self.prepare_data(data)
        y = data['landslide_occurred'].values  # FIX: Changed from flood_occurred
        
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
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
        
        # Store results
        training_result = {
            'timestamp': datetime.now(),
            'algorithm': self.algorithm,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': np.mean(cv_scores) if len(cv_scores) > 0 else None,  # ✅ Fixed
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
        """Predict landslide risk level for a specific location"""
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
        
        if probability < 0.25:
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