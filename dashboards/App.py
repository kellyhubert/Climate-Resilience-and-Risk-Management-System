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

