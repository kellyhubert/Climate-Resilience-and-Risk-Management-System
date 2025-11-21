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
import os
import random
import math
from datetime import datetime, timedelta
import requests

# Try to import numpy (optional)
try:
    import numpy as np
    numpy_available = True
except ImportError:
    numpy_available = False
    print("[WARNING] numpy not available - using Python random module")
    # Create numpy-like interface using Python's random module
    class np:
        @staticmethod
        def random():
            class RandomModule:
                @staticmethod
                def choice(options, p=None):
                    if p:
                        return random.choices(options, weights=p, k=1)[0]
                    return random.choice(options)

                @staticmethod
                def uniform(low, high):
                    return random.uniform(low, high)

                @staticmethod
                def gamma(shape, scale):
                    # Simple gamma approximation
                    return random.gammavariate(shape, scale)
            return RandomModule()

        @staticmethod
        def sin(x):
            return math.sin(x)

        @staticmethod
        def pi():
            return math.pi

        random = random

    # Replace np.random with our mock
    np.random = np.random()
    np.pi = math.pi

# Try to import pandas (optional)
try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    print("[WARNING] pandas not available - some features may be limited")

# Configuration
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', "c45952d88bbd1eae279947148383f7d6")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Rwanda's 30 Districts with coordinates (approximate district centers)
RWANDA_DISTRICTS = {
    # Eastern Province
    'Bugesera': {'lat': -2.2167, 'lon': 30.2167, 'province': 'Eastern'},
    'Gatsibo': {'lat': -1.6333, 'lon': 30.4000, 'province': 'Eastern'},
    'Kayonza': {'lat': -1.8833, 'lon': 30.6333, 'province': 'Eastern'},
    'Kirehe': {'lat': -2.2167, 'lon': 30.7000, 'province': 'Eastern'},
    'Ngoma': {'lat': -2.1667, 'lon': 30.5333, 'province': 'Eastern'},
    'Nyagatare': {'lat': -1.3000, 'lon': 30.3333, 'province': 'Eastern'},
    'Rwamagana': {'lat': -1.9500, 'lon': 30.4333, 'province': 'Eastern'},

    # Kigali Province
    'Gasabo': {'lat': -1.9167, 'lon': 30.1000, 'province': 'Kigali'},
    'Kicukiro': {'lat': -1.9833, 'lon': 30.1000, 'province': 'Kigali'},
    'Nyarugenge': {'lat': -1.9500, 'lon': 30.0500, 'province': 'Kigali'},

    # Northern Province
    'Burera': {'lat': -1.4833, 'lon': 29.8833, 'province': 'Northern'},
    'Gakenke': {'lat': -1.6667, 'lon': 29.7667, 'province': 'Northern'},
    'Gicumbi': {'lat': -1.6333, 'lon': 30.0167, 'province': 'Northern'},
    'Musanze': {'lat': -1.4994, 'lon': 29.6338, 'province': 'Northern'},
    'Rulindo': {'lat': -1.7667, 'lon': 30.0833, 'province': 'Northern'},

    # Southern Province
    'Gisagara': {'lat': -2.5667, 'lon': 29.8333, 'province': 'Southern'},
    'Huye': {'lat': -2.6067, 'lon': 29.7394, 'province': 'Southern'},
    'Kamonyi': {'lat': -2.0833, 'lon': 29.8833, 'province': 'Southern'},
    'Muhanga': {'lat': -2.0833, 'lon': 29.7333, 'province': 'Southern'},
    'Nyamagabe': {'lat': -2.5333, 'lon': 29.4333, 'province': 'Southern'},
    'Nyanza': {'lat': -2.3500, 'lon': 29.7500, 'province': 'Southern'},
    'Nyaruguru': {'lat': -2.6333, 'lon': 29.3833, 'province': 'Southern'},
    'Ruhango': {'lat': -2.2333, 'lon': 29.7833, 'province': 'Southern'},

    # Western Province
    'Karongi': {'lat': -2.0000, 'lon': 29.3667, 'province': 'Western'},
    'Ngororero': {'lat': -1.8167, 'lon': 29.5833, 'province': 'Western'},
    'Nyabihu': {'lat': -1.6333, 'lon': 29.5000, 'province': 'Western'},
    'Nyamasheke': {'lat': -2.3500, 'lon': 29.0833, 'province': 'Western'},
    'Rubavu': {'lat': -1.6500, 'lon': 29.3000, 'province': 'Western'},
    'Rusizi': {'lat': -2.4843, 'lon': 28.9086, 'province': 'Western'},
    'Rutsiro': {'lat': -1.9667, 'lon': 29.3333, 'province': 'Western'},
}

# Add parent directory to path to import models
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

models_available = False
try:
    from models.flood_model import RwandaFloodModel
    from models.drought_model import RwandaDroughtModel
    from models.landslide_model import RwandaLandslideModel 
    from models.base_model import BaseRiskModel
    models_available = True
    print("[OK] Model classes imported successfully")
except ImportError as e:
    print(f"[WARNING] Could not import models: {e}")
    print("Running in demo mode")
    RwandaLandslideModel = None
    RwandaFloodModel = None
    RwandaDroughtModel = None
    BaseRiskModel = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Configuration
app.config['SECRET_KEY'] = 'rwanda-climate-risk-2024'
app.config['MODELS_DIR'] = project_root / 'models' / 'trained'  # Fixed typo

# Global variables for loaded models
loaded_models = {
    'landslide': None,
    'flood': None,
    'drought': None,
    'ensemble': None
}

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load trained models at startup"""
    global loaded_models
    
    if not models_available:
        print("[WARNING] Models not available - running in demo mode")
        return
    
    print("\n" + "=" * 60)
    print("Loading trained models...")
    print("=" * 60)
    
    models_dir = app.config['MODELS_DIR']
    
    if not models_dir.exists():
        print(f"[ERROR] Models directory not found: {models_dir}")
        print("Please run: python scripts/train_all_models.py")
        return
    
    # Load Landslide Model
    landslide_path = models_dir / 'rwanda_landslide_model.pkl'
    if landslide_path.exists():
        try:
            loaded_models['landslide'] = RwandaLandslideModel()
            loaded_models['landslide'].load_model(landslide_path)
            print("[OK] Landslide model loaded")
        except Exception as e:
            print(f"[ERROR] Error loading landslide model: {e}")
    else:
        print(f"[WARNING] Landslide model not found: {landslide_path}")
    
    # Load Flood Model
    flood_path = models_dir / 'rwanda_flood_model.pkl'
    if flood_path.exists():
        try:
            loaded_models['flood'] = RwandaFloodModel()
            loaded_models['flood'].load_model(flood_path)
            print("[OK] Flood model loaded")
        except Exception as e:
            print(f"[ERROR] Error loading flood model: {e}")
    else:
        print(f"[WARNING] Flood model not found: {flood_path}")
    
    # Load Drought Model
    drought_path = models_dir / 'rwanda_drought_model.pkl'
    if drought_path.exists():
        try:
            loaded_models['drought'] = RwandaDroughtModel()
            loaded_models['drought'].load_model(drought_path)
            print("[OK] Drought model loaded")
        except Exception as e:
            print(f"[ERROR] Error loading drought model: {e}")
    else:
        print(f"[WARNING] Drought model not found: {drought_path}")
    
    # Summary
    models_loaded = sum(1 for m in loaded_models.values() if m is not None)
    print("=" * 60)
    print(f"Models loaded: {models_loaded}/3")
    print("=" * 60 + "\n")

# ============================================================================
# DEMO DATA GENERATION
# ============================================================================

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
    """Generate demo weather data with time-based variation"""
    stations_info = {
        'Kigali Airport': {'base_temp': 22, 'base_rain': 10},
        'Butare': {'base_temp': 20, 'base_rain': 12},
        'Ruhengeri': {'base_temp': 18, 'base_rain': 15},
        'Cyangugu': {'base_temp': 23, 'base_rain': 14},
        'Kibungo': {'base_temp': 24, 'base_rain': 8}
    }
    
    weather = []
    current_hour = datetime.now().hour
    
    for station, info in stations_info.items():
        # Temperature varies by time of day
        temp_variation = 5 * np.sin((current_hour - 14) * np.pi / 12)
        
        weather.append({
            'station': station,
            'temperature': round(info['base_temp'] + temp_variation + np.random.uniform(-2, 2), 1),
            'humidity': round(np.random.uniform(60, 95), 0),
            'rainfall_24h': round(info['base_rain'] + np.random.gamma(1, 3), 1),
            'wind_speed': round(np.random.uniform(0, 15), 1),
            'timestamp': datetime.now().isoformat()
        })
    
    return weather

# ============================================================================
# ROUTES - PAGES
# ============================================================================

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

# ============================================================================
# API ROUTES - DATA
# ============================================================================

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
    """Get current weather data (demo mode)"""
    weather = generate_demo_weather()
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'stations': weather,
        'source': 'Demo Data'
    })

# NEW ROUTE: REAL-TIME WEATHER
@app.route('/api/weather-realtime')
def api_weather_realtime():
    """Get real-time weather from OpenWeatherMap API"""
    
    # Rwanda weather stations with coordinates
    stations = {
        'Kigali Airport': {'lat': -1.9706, 'lon': 30.1394},
        'Butare': {'lat': -2.6067, 'lon': 29.7394},
        'Ruhengeri': {'lat': -1.4994, 'lon': 29.6338},
        'Cyangugu': {'lat': -2.4843, 'lon': 28.9086},
        'Kibungo': {'lat': -2.1534, 'lon': 30.7677}
    }
    
    weather_data = []
    
    # Check if API key is configured
    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        print("[WARNING] OpenWeatherMap API key not configured. Using demo data.")
        return api_weather()  # Fallback to demo data
    
    for station_name, coords in stations.items():
        try:
            # Call OpenWeatherMap API
            response = requests.get(
                OPENWEATHER_BASE_URL,
                params={
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'appid': OPENWEATHER_API_KEY,
                    'units': 'metric'  # Celsius
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                weather_data.append({
                    'station': station_name,
                    'temperature': round(data['main']['temp'], 1),
                    'humidity': data['main']['humidity'],
                    'rainfall_24h': data.get('rain', {}).get('1h', 0) * 24,  # Estimate 24h from 1h
                    'wind_speed': round(data['wind']['speed'] * 3.6, 1),  # Convert m/s to km/h
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'timestamp': datetime.now().isoformat()
                })
            else:
                print(f"API error for {station_name}: {response.status_code}")
                raise Exception("API request failed")
                
        except Exception as e:
            print(f"Error fetching weather for {station_name}: {e}")
            # Fallback to demo data for this station
            weather_data.append({
                'station': station_name,
                'temperature': round(20 + np.random.uniform(-5, 10), 1),
                'humidity': round(np.random.uniform(60, 95), 0),
                'rainfall_24h': round(np.random.gamma(2, 8), 1),
                'wind_speed': round(np.random.uniform(0, 15), 1),
                'timestamp': datetime.now().isoformat(),
                'note': 'Demo data - API unavailable'
            })
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'stations': weather_data,
        'source': 'OpenWeatherMap API' if len(weather_data) > 0 else 'Demo Data'
    })

# NEW ROUTE: WEATHER FORECAST FOR ALL 30 DISTRICTS
@app.route('/api/weather-forecast')
def api_weather_forecast():
    """Get weather forecast for all 30 districts in Rwanda"""

    # Check if API key is configured
    if OPENWEATHER_API_KEY == "YOUR_API_KEY_HERE":
        print("[WARNING] OpenWeatherMap API key not configured.")
        return jsonify({
            'status': 'error',
            'message': 'API key not configured'
        }), 500

    forecast_data = []

    for district_name, coords in RWANDA_DISTRICTS.items():
        try:
            # Call OpenWeatherMap Forecast API (5-day forecast in 3-hour intervals)
            response = requests.get(
                OPENWEATHER_FORECAST_URL,
                params={
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'appid': OPENWEATHER_API_KEY,
                    'units': 'metric',
                    'cnt': 8  # Get next 24 hours (8 x 3-hour intervals)
                },
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                # Process forecast data
                forecasts = []
                weather_conditions = []

                for item in data['list']:
                    forecast_time = datetime.fromtimestamp(item['dt'])
                    weather_main = item['weather'][0]['main'].lower()
                    weather_desc = item['weather'][0]['description']

                    weather_conditions.append(weather_main)

                    forecasts.append({
                        'time': forecast_time.strftime('%Y-%m-%d %H:%M'),
                        'temp': round(item['main']['temp'], 1),
                        'condition': weather_main,
                        'description': weather_desc,
                        'humidity': item['main']['humidity'],
                        'rain_probability': item.get('pop', 0) * 100  # Probability of precipitation
                    })

                # Determine overall weather prediction (sunny, cloudy, or rain)
                rain_conditions = ['rain', 'drizzle', 'thunderstorm']
                cloud_conditions = ['clouds']

                rain_count = sum(1 for cond in weather_conditions if cond in rain_conditions)
                cloud_count = sum(1 for cond in weather_conditions if cond in cloud_conditions)

                if rain_count > len(weather_conditions) * 0.3:  # More than 30% rain
                    prediction = 'rain'
                    icon = '🌧️'
                elif cloud_count > len(weather_conditions) * 0.5:  # More than 50% cloudy
                    prediction = 'cloudy'
                    icon = '☁️'
                else:
                    prediction = 'sunny'
                    icon = '☀️'

                forecast_data.append({
                    'district': district_name,
                    'province': coords['province'],
                    'prediction': prediction,
                    'icon': icon,
                    'forecasts': forecasts[:4],  # Return next 4 time periods (12 hours)
                    'avg_temp': round(sum(f['temp'] for f in forecasts) / len(forecasts), 1),
                    'max_temp': round(max(f['temp'] for f in forecasts), 1),
                    'min_temp': round(min(f['temp'] for f in forecasts), 1)
                })
            else:
                print(f"API error for {district_name}: {response.status_code}")

        except Exception as e:
            print(f"Error fetching forecast for {district_name}: {e}")
            # Add placeholder data for failed requests
            forecast_data.append({
                'district': district_name,
                'province': coords['province'],
                'prediction': 'unknown',
                'icon': '❓',
                'error': 'Data unavailable'
            })

    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'total_districts': len(forecast_data),
        'forecasts': sorted(forecast_data, key=lambda x: (x['province'], x['district'])),
        'source': 'OpenWeatherMap Forecast API'
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make risk prediction for given features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Check if any models are loaded
        if not any(loaded_models.values()):
            return jsonify({
                'status': 'error',
                'message': 'Models not loaded. Please train models first: python scripts/train_all_models.py'
            }), 503
        
        results = {}

        # Landslide prediction
        if loaded_models['landslide']:
            try:
                results['landslide'] = loaded_models['landslide'].predict_risk_level(data)
            except Exception as e:
                results['landslide'] = {'error': f'Landslide prediction failed: {str(e)}'}
        else:
            results['landslide'] = {'error': 'Landslide model not loaded'}

        # Flood prediction
        if loaded_models['flood']:
            try:
                results['flood'] = loaded_models['flood'].predict_risk_level(data)
            except Exception as e:
                results['flood'] = {'error': f'Flood prediction failed: {str(e)}'}
        else:
            results['flood'] = {'error': 'Flood model not loaded'}

        # Drought prediction
        if loaded_models['drought']:
            try:
                results['drought'] = loaded_models['drought'].predict_risk_level(data)
            except Exception as e:
                results['drought'] = {'error': f'Drought prediction failed: {str(e)}'}
        else:
            results['drought'] = {'error': 'Drought model not loaded'}
        
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
        {'name': 'Musanze', 'province': 'Northern', 'population': 476520},
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

# ============================================================================
# ERROR HANDLERS
# ============================================================================

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

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Rwanda Climate Risk Early Warning System - Dashboard")
    print("=" * 60)
    
    # Create necessary directories
    templates_dir = Path(__file__).parent / 'templates'
    static_dir = Path(__file__).parent / 'static'
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)

    # Load models
    if models_available:
        load_models()
    else:
        print("[WARNING] Running in demo mode - models not available")
    
    print("\nStarting Flask server...")
    print("Dashboard: http://localhost:5000")
    print("API Status: http://localhost:5000/api/status")
    print("API Weather (Demo): http://localhost:5000/api/weather")
    print("API Weather (Real-time): http://localhost:5000/api/weather-realtime")
    print("\nPress Ctrl+C to stop\n")

    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )