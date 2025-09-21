"""Central confugguration for rwanda climate risk system."""

import os
from pathlib import Path

class Config:
    """Main configuration class"""

    #project paths

    PROJECT_ROOT= Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

    #Data sub-directories
    RAW_DATA = DATA_DIR / "raw"
    PROCESSED_DATA = DATA_DIR / "processed"
    CLIMATE_DATA = DATA_DIR / "climate"
    GEO_DATA = DATA_DIR / "geographic"
    SOCIO_DATA = DATA_DIR / "socioeconomic"

    # Rwanda-specifica settings
    RWANDA_BOUNDS = {
        'min_lat': -2.94, 'max_lat': -1.05,
        'min_lon': 28.86, 'max_lon': 30.90
    }
    

    #Risk thresholds

    LANDSLIDE_RAIN_THRESHOLD = 50  # mm in 24 hours
    FLOOD_RAIN_THRESHOLD = 75  # mm in 24 hours
    DROUGHT_SPI_THRESHOLD = -1.5  # Standardized Precipitation Index

    # Alert settings
    SMS_API_KEY = os.getenv("SMS_API_KEY", "your_default_api_key")
    EMAIL_SMTP=os.getenv("EMAIL_SMTP", "smtp.gmail.com")
    
    #DAtabase settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///climate_risk.db")
    # DATABASE_USER = os.getenv("DATABASE_USER", "user")
    # DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "password")
    
    #API endpoints
    WEATHER_API_URL = "http: //api.meteo.gov.rw"
    SATELLITE_API_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"