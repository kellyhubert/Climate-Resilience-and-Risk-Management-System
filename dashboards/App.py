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

#Routes Pages

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/map')
def risk_map():
    """Interactive risk map page"""
    return render_template('map.html')

@app.route('/alerts')
def alerts_page():
    """Alerts management page"""
    return render_template('alerts.html')

@app.route('/analytics')
def analytics():
    """Analytics and reports page"""
    return render_template('analytics.html')

@app.route('/predict')
def predict_page():
    """Risk prediction tool page"""
    return render_template('predict.html')

#Demo Data
@app.route('/api/status')
def api_status():
    """Get system status"""
    return jsonify({
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': {
            'landslide': loaded_models['landslide'] is not None,
            'flood': loaded_models['flood'] is not None,
            'drought': loaded_models['drought'] is not None
        },
        'version': '1.0.0'
    })

@app.route('/api/current-risks')
def api_current_risks():
    """Get current risk levels for all districts"""
    risks = generate_demo_risk_data()
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'data': risks,
        'summary': {
            'high_risk_districts': len([r for r in risks if r['landslide_risk'] in ['High', 'Critical']]),
            'total_districts': len(risks)
        }
    })

@app.route('/api/alerts')
def api_alerts():
    """Get current active alerts"""
    alerts = generate_demo_alerts()
    
    # Filter by severity if requested
    severity = request.args.get('severity')
    if severity:
        alerts = [a for a in alerts if a['severity'] == severity]
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'count': len(alerts),
        'alerts': alerts
    })

@app.route('/api/weather')
def api_weather():
    """Get current weather data"""
    weather = generate_demo_weather()
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'stations': weather
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make risk prediction for given features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        results = {}

        # Landslide prediction
        if 'elevation_m' in data and loaded_models['landslide']:
            try:
                results['landslide'] = loaded_models['landslide'].predict_risk_level(data)
            except Exception as e:
                results['landslide'] = {'error': str(e)}

        # Flood prediction
        if 'rainfall_1h_mm' in data and loaded_models['flood']:
            try:
                results['flood'] = loaded_models['flood'].predict_risk_level(data)
            except Exception as e:
                results['flood'] = {'error': str(e)}

        # Drought prediction
        if 'spi_3month' in data and loaded_models['drought']:
            try:
                results['drought'] = loaded_models['drought'].predict_risk_level(data)
            except Exception as e:
                results['drought'] = {'error': str(e)}
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': 'Insufficient data for prediction. Models not loaded or required features missing.'
            }), 400
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/api/historical-data')
def api_historical_data():
    """Get historical risk data for charts"""
    # Generate sample time series data
    days = 30
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days, 0, -1)]
    
    data = {
        'dates': dates,
        'landslide_risk': [np.random.uniform(0.2, 0.8) for _ in range(days)],
        'flood_risk': [np.random.uniform(0.1, 0.7) for _ in range(days)],
        'drought_risk': [np.random.uniform(0.15, 0.6) for _ in range(days)],
        'rainfall': [np.random.gamma(2, 10) for _ in range(days)]
    }
    
    return jsonify({
        'status': 'success',
        'data': data
    })

@app.route('/api/districts')
def api_districts():
    """Get list of all districts with metadata"""
    districts = [
        {'name': 'Kigali', 'province': 'Kigali City', 'population': 1329830},
        {'name': 'Gasabo', 'province': 'Kigali City', 'population': 529561},
        {'name': 'Kicukiro', 'province': 'Kigali City', 'population': 491731},
        {'name': 'Nyarugenge', 'province': 'Kigali City', 'population': 374319},
        {'name': 'Musanze', 'province': 'Northern', 'population':  476520},
        {'name': 'Burera', 'province': 'Northern', 'population': 387729},
        {'name': 'Gicumbi', 'province': 'Northern', 'population': 448824},
        {'name': 'Nyaruguru', 'province': 'Southern', 'population': 318126},
        {'name': 'Kirehe', 'province': 'Eastern', 'population': 460860},
        {'name': 'Rubavu', 'province': 'Western', 'population': 407406}
    ]
    
    return jsonify({
        'status': 'success',
        'count': len(districts),
        'districts': districts
    })

@app.route('/api/statistics')
def api_statistics():
    """Get system statistics"""
    stats = {
        'total_predictions': np.random.randint(1000, 5000),
        'alerts_generated_today': np.random.randint(5, 20),
        'districts_monitored': 30,
        'weather_stations': 5,
        'model_accuracy': {
            'landslide': 0.87,
            'flood': 0.84,
            'drought': 0.82
        },
        'last_update': datetime.now().isoformat()
    }
    
    return jsonify({
        'status': 'success',
        'statistics': stats
    })

#Error Handlers

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500