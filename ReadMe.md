Rwanda Climate Resilience and Risk Management System

""An advanced machine learning-based early warning system for predicting and monitoring climate-related risks in Rwanda, including landslides, floods, and droughts.""

# Table of Contents

**Overview
*Features
*Project Structure
*Installation
*Usage
*Models
*API Documentation
*Dashboard
*Technologies
*Contributing
*License*

 ## Overview

The Rwanda Climate Resilience and Risk Management System is a comprehensive platform designed to:

**Predict climate risks** using machine learning models trained on Rwanda-specific geographic and meteorological data.
**Provide real-time monitoring** through an interactive web dashboard
**Generate early warnings** for landslides, floods, and droughts
**Support decision-making** for disaster preparedness and response

### Target Audience

Government officials and disaster management agencies
Climate researchers and scientists
NGOs working in disaster risk reduction
Local communities and policymakers
 
# Features

 ### Machine Learning Models

**Landslide Risk Prediction:** Analyzes terrain, rainfall, soil conditions, and vegetation
**Flood Risk Prediction:** Evaluates rainfall patterns, drainage systems, and urbanization
**Drought Risk Prediction:** Monitors rainfall deficits, temperature anomalies, and vegetation indices

### Interactive Dashboard
- Real-time risk visualization for all districts
- Interactive risk maps with geographic overlays
- Weather station monitoring
- Active alerts management
- Historical data analytics
- Custom prediction tool with user inputs

### Early Warning System
- Automated risk level classification (Low, Medium, High, Critical)
- District-level risk assessments
- Real-time weather data integration
- Alert generation based on threshold exceedance

### REST API

- RESTful endpoints for programmatic access
- JSON responses for easy integration
- Real-time predictions and historical data
- Status monitoring and health checks


# Installation
 
### Prerequisites
- Python 3.8 or higher  
- pip (Python package manager)  
- Git  

### Step 1: Clone the Repository

`git clone https://github.com/yourusername/Climate-Resilience-and-Risk-Management-System.git
cd Climate-Resilience-and-Risk-Management-System`

### Step 2: Create Virtual Environment (Recommended)
`# Windows
python -m venv venv
venv\Scripts\activate`

`# Linux/Mac
python3 -m venv venv
source venv/bin/activate`

### Step 3: Install Dependencies
`pip install -r requirements.txt`

### Step 4: Train the Models

`python scripts/train_all_models.py`

**This will:**

Generate synthetic training data (4000-6000 samples per model)
Train Random Forest and Gradient Boosting models
Save trained models to **models/trained/**
Display training accuracy and cross-validation scores

### Step 5: Start the Dashboard
`cd dashboards
python App.py`

The dashboard will be available at: http://localhost:5000

## Usage
### Quick Start

**1 Access the Dashboard**

Open your browser and navigate to http://localhost:5000


**2 View Current Risks**

The main dashboard shows risk levels for all districts
Color-coded risk indicators (green=low, yellow=medium, orange=high, red=critical)


**3 Make Predictions**

Navigate to the "Predict" page
Enter location parameters (elevation, slope, rainfall, etc.)
Click "Predict Risk" to get instant predictions for all three hazards


**4 Monitor Alerts**

View active alerts on the "Alerts" page
Filter by severity or hazard type


**5 Analyze Data**

Access historical trends on the "Analytics" page
View charts and statistics

### Using the Prediction Tool
The prediction tool accepts the following inputs:
**Geographic Features:**

Elevation (meters): 900-4500m
Slope (degrees): 0-60°
Distance to river (meters)

**Meteorological Features:**

1-hour rainfall (mm)
24-hour rainfall (mm)
Soil moisture (0-1)

**Land Use:**

Forest, Agriculture, Urban, or Grassland

**Location/Region:**

Northern Mountains, Central Plateau, Eastern Hills, Western Ridges, Southern Valleys

## Models
**1. Landslide Risk Model**
Algorithm: Random Forest Classifier (200 estimators)
**Key Features:**

Slope angle (most important)
Rainfall intensity (1h, 6h, 24h)
Soil saturation
Vegetation cover (NDVI)
Urbanization degree
Drainage conditions

**Risk Factors:**

Steep slopes (>25°)
Heavy rainfall (>50mm/24h)
Saturated soils (>80%)
Poor vegetation cover
High urbanization with poor drainage


**2. Flood Risk Model**
**Algorithm:** Gradient Boosting Classifier
**Key Features:**

Rainfall intensity (critical factor)
Proximity to water bodies
Topography (elevation, slope)
Urban drainage capacity
Impervious surface percentage

**Risk Factors:**

Low elevation areas
Flat terrain (slope <10°)
Near rivers/lakes
High urbanization
Poor drainage systems


**3. Drought Risk Model**
**Algorithm:** Random Forest Classifier (250 estimators)
Key Features:

Standardized Precipitation Index (SPI)
Rainfall deficit
Temperature anomalies
Vegetation stress (NDVI)
Soil moisture deficit
Consecutive dry days

**Risk Factors:**

SPI < -1.0 (meteorological drought)
Rainfall deficit >30%
High temperatures (>28°C)
Low vegetation health
Extended dry periods (>30 days)

## API Documentation
### Base URL
`http://localhost:5000/api`

### Endpoints
**1. System Status**
`GET /api/status`

**2. Make Prediction**
`POST /api/predict
Content-Type: application/json`

**3. Get Current Risks**
`GET /api/current-risks`
Returns risk levels for all districts.

**4. Get Active Alerts**
`GET /api/alerts`

**5. Get Weather Data**
`GET /api/weather`

**6. Get Historical Data**
`GET /api/historical-data`
Returns 30-day historical risk trends.

## Dashboard
### Pages

**Main Dashboard (/)**

Overview of current risk levels
Summary statistics
Recent alerts
Quick access to all features


**Interactive Map (/map)
**
Geographic visualization of risks
District-level risk overlays
Click districts for detailed information


**Alerts (/alerts)**

List of active alerts
Filter by type and severity
Historical alert archive


**Analytics (/analytics)**

Historical trends and charts
Statistical analysis
Model performance metrics

**Prediction Tool (/predict)**

Custom risk prediction
Interactive form with validation
Instant results for all hazards

## Technologies
### Backend

**Flask:** Web framework
**scikit-learn:** Machine learning
**NumPy & Pandas:** Data processing
**Pickle/Joblib:** Model serialization

### Frontend

**HTML5/CSS3:** UI structure and styling
**JavaScript: **Interactive features
**Responsive Design:** Mobile-friendly interface

### Machine Learning

**Random Forest:** Landslide and Drought models
**Gradient Boosting**: Flood model
**Feature Engineering:** Domain-specific features
**Cross-validation:** Model evaluation

### Data Processing

**StandardScaler**: Feature normalization
**LabelEncoder:** Categorical encoding
**SelectKBest:** Feature selection


## Future Enhancements

- Integration with real-time satellite data
- Mobile application for field officers
- SMS/Email alert notifications
- Multi-language support (Kinyarwanda, French, English)
- Advanced ensemble models
- Real-time weather API integration
- Historical event database
- Community reporting features
- Export reports to PDF
- Integration with national disaster management systems

##  Authors

Kelly Hubert Irakoze- initial-Work-kellyhubert(GitHub)

## Contact

Email: irakellyhub@gmail.com
GitHub Issues: https://github.com/kellyhubert