class RwandaLAndslideModel(BaseRiskModel):
    
    """Advance Landslide Risk prediction Model for Rwanda."""
    def __init__(self, model_algorithm='RandomForest'):
        super().__init__("Rwanda_Landslde_Model", "classification")
        self.algorithm = model_algorithm
        self.rwanda_params = self.__setup_rwanda_parameters()
        self.__initialize_model()

    def _setup_rwanda_parameters(self):
        """Rwanda specific landslide parameters."""
        return {
             'elevation_range': (900,4507),
             'critical_slope': 25,
             'rainfall_threshold': 50,
             'soil_saturation_threshold': 0.8,
             'vegetation_threshold': 0.3,
             'geology_weighht':{
                  'volcanic':0.6,
                  'sedimentary':0.8,
                  'weathered':1.0,
                  'alluvial':0.7
             }
        }

       
    def _initialize_model(self):
        """Initialize the underlying ML model"""
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
                )
        elif self.algorithim == 'gradient_boosting':
                self.model = GradientBoostingClassifier(

                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                )
        elif self.algorithm == 'logistic_regression':
             self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
             )
        
    def generate_training_data(self, n_samples = 5000):
         """Generate realistic training dtata for Rwanda landslides"""
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
              #select random region
              region_name, region_params = np.random.choice(list(regions.items()))

              elevation = np.random.normal(region_params['elevation_mean'], 200)
              elevation = np.clip(elevation, 900, 4500)

              slope = np.random.normal(region_params['elevation_mean'], 8)
              slope = np.clip(slope, 0, 60)

              aspect = np.random.uniform(0, 360)

              #Generate cllimate features
              #Rainfall patteern (more in wet season)

              season_factors = np.SIN((i % 365) * 2 * np.pi / 365) * 0.3 + 1  # Seasonal variation
              daily_rainfall = np.random.gamma(2, 3) * season_factors
              rainfall_24h = daily_rainfall + np.random.gamma(1,10) # Extreme event
              rainfall_72h = rainfall_24h + np.random.gamma(1,15)
              anatecendent_rainfall = np.random.gamma(3, 8) #Previoous week rainfall

              