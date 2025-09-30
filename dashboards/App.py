"""
Rwanda Climate Risk Early Warning System - Web Dashboard.
This Flask application provides a web interface for monitoring climate risks,
viewing alerts, and accessing risk predictions for Rwanda.
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.flood_model import RwandaFloodModel
    from models.drought_model import RwandaDroughtModel
    from models.landslide_model import RwandaLAndslideModel
    from models.ensemble_model import RwandaEnsembleModel
    from models.base_model import BaseRiskModel
except ImportError:
    print("Warning: Could not import models. Running in demo mode.")

    RwandaLAndslideModel = None
    RwandaFloodModel = None
    RwandaDroughtModel = None
    RwandaEnsembleModel = None
    BaseRiskModel = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# configuration
app.config['SECRET_KEY'] = 'rwanda-climate-risk-2024'
app.config['MODELS_DI'] = Path(__file__).parent.parent / 'models'

#Global variables for loaded models
loaded_models = {
    'landslide': None,
    'flood': None,
    'drought': None,
    'ensemble': None
}

#Model Loading
def load_models():
    """Load trained models at startup"""
    global loaded_models
    
    print("Loading models...")
    models_dir = app.config['MODELS_DIR']
    
    try:
        # Try to load pre-trained models
        if (models_dir / 'rwanda_landslide_model.pkl').exists():
            loaded_models['landslide'] = RwandaLAndslideModel()
            loaded_models['landslide'].load_model(models_dir / 'rwanda_landslide_model.pkl')
            print("✓ Landslide model loaded")
        
        if (models_dir / 'rwanda_flood_model.pkl').exists():
            loaded_models['flood'] = RwandaFloodModel()
            loaded_models['flood'].load_model(models_dir / 'rwanda_flood_model.pkl')
            print("✓ Flood model loaded")
        
        if (models_dir / 'rwanda_drought_model.pkl').exists():
            loaded_models['drought'] = RwandaDroughtModel()
            loaded_models['drought'].load_model(models_dir / 'rwanda_drought_model.pkl')
            print("✓ Drought model loaded")
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Running in demo mode with simulated data")

#Demo data Generation
def generate_demo_risk_data():
        """Generate demo risk data for visualization"""
        districts = [
        'Kigali', 'Gasabo', 'Kicukiro', 'Nyarugenge',
        'Musanze', 'Burera', 'Gicumbi', 'Rulindo',
        'Nyaruguru', 'Gisagara', 'Nyamagabe', 'Muhanga',
        'Kirehe', 'Gatsibo', 'Kayonza', 'Ngoma',
        'Rubavu', 'Nyabihu', 'Ngororero', 'Rusizi'
         ]
        risks = []
        for district in districts:
            risks.append({
            'district': district,
            'landslide_risk': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                              p=[0.5, 0.3, 0.15, 0.05]),
            'flood_risk': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                          p=[0.4, 0.35, 0.2, 0.05]),
            'drought_risk': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 
                                            p=[0.6, 0.25, 0.1, 0.05]),
            'landslide_prob': np.random.uniform(0.1, 0.9),
            'flood_prob': np.random.uniform(0.1, 0.8),
            'drought_prob': np.random.uniform(0.1, 0.7)
        })
    
        return risks

def generate_demo_alerts():
    """Generate demo alerts"""
    alerts = [
        {
            'id': 'LS_001',
            'type': 'landslide',
            'severity': 'High',
            'district': 'Musanze',
            'message': 'Heavy rainfall detected. High landslide risk in mountainous areas.',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'affected_population': 15000
        },
        {
            'id': 'FL_002',
            'type': 'flood',
            'severity': 'Medium',
            'district': 'Kigali',
            'message': 'Urban flooding possible in low-lying areas. Monitor Nyabugogo wetlands.',
            'timestamp': (datetime.now() - timedelta(hours=5)).isoformat(),
            'affected_population': 25000
        },
        {
            'id': 'DR_003',
            'type': 'drought',
            'severity': 'Medium',
            'district': 'Kirehe',
            'message': 'Below-average rainfall for Season C. Implement water conservation.',
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
            'affected_population': 50000
        }
    ]
    
    return alerts

def generate_demo_weather():
    """Generate demo weather data"""
    stations = ['Kigali Airport', 'Butare', 'Ruhengeri', 'Cyangugu', 'Kibungo']
    
    weather = []
    for station in stations:
        weather.append({
            'station': station,
            'temperature': round(20 + np.random.uniform(-5, 10), 1),
            'humidity': round(np.random.uniform(60, 95), 0),
            'rainfall_24h': round(np.random.gamma(2, 8), 1),
            'wind_speed': round(np.random.uniform(0, 15), 1),
            'timestamp': datetime.now().isoformat()
        })
    
    return weather
