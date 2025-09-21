"""Data loading utilities for Rwanda climate risk system."""

import pandas as pd
import geopandas as gpd
import rasterio
import json
from pathlib import Path
from config import Config

class DataLoader:
    """Handles loading all types of data."""

    def __init__(self):
        self.config = Config()

    def load_weather_data(self, filename=None):
        # """Load weather data from CSV."""
        if filename is None:
            filename = self.config.CLIMATE_DATA / "weather_data.csv"

        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except FileNotFoundError:
            print(f"Weather data file not found: {filename}")
            return self.create_sample_weather_data()
    
    def load_admin_boundaries(self, level='district'):
        """Load Rwanda administrative boundaries."""

        filename = self.config.GEO_DATA / f"rwanda_{level}s.geojson"

        try:
            gdf= gdf.read_file(filename)
            return gdf
        except FileNotFoundError:
            print(f"Weather data not found: {filename}")
            return self.create_sample_boundaries()
        
    def load_elevtion_data(self):
            """"Load elevation rasta data."""

            filename= self.config.GEO_DATA / "rwanda_elevation.tif"

            try:
                with rasterio.open(filename) as src:
                    elevation = src.read(1)
                    bounds = src.bounds
                    crs = src.crs 
                return elevation, bounds, crs
            except FileNotFoundError:
                print(f"Elevation data  not found: {filename}")
                return None, None, None
        
    def load_vulnarability_data(self):
            """Load community vulnerability data"""

            filename = self.config.SOCIO_DATA / "vulnerability_index.csv"
            try:
                df = pd.read_csv(filename)
                return df
            except FileNotFoundError:
                print(f"Vulnerabilty data not found: {filename}")
                return self.create_sample_vulnerability_data()
            
    def create_sample_weather_data(self):
            """Create sample weather data for testing."""

            print("Createing sample weather data...")
            import numpy as np
            from datetime import datetime, timedelta

             # Create & days pof hourly weather data for 5 stations

            dates= pd.date_range(
                start=datetime.now() - timedelta(days=7),
                end = datetime.now(),
                freq='H'
                )
            
            stations=['Kigali','Butare','Gisenyi','Huye','Musanze']
            data = []
            for station in staions:
                for date in dates:
                    record= {
                        'station': station,
                        'timestamp': date,
                        'temperature':np.random.normal(27, 18),
                        'humidity': np.random.uniform(40, 90),
                        'rainfall':np.random.gamma(0.5,2),
                        'pressure': np.random.normal(1013, 10)
                    }
                    data.append(record)
            df = pd.DataFrame(data)

            #save sample data
            output_file = self.config.CLIMATE_DATA / "sample_weather_data.csv"
            df.to_csv(output_file, index=False)
            print(f"Sample weather data saved to {output_file}")

            return df
        
    def create_sample_boundaries(self):
            """"Create sample district boundaries"""

            print("Createing sample boundary data...")

            #Rwanda disttricts (Simplified)

            districts_data ={
                'district': ['Kigali', 'Musanze','Huye', 'Rwamagana','Rubavu'],
                'province': ['Kigali', 'Northern','Southern','Eastern','Western'],
                'area_km2': [730, 530, 582, 682, 388],
                'population': [1745555,476522,381900,484953,546683]
            }

            from shapely.geometry import box
            import geopandas as gpd
            import numpy as np
            geometries = []
            lats= np.llnspace(-1.9, -1.2, 5)
            lons = np.linspace(29.0, 30.5, 5)

            for i in range(5):
                 
                 #create a simple rectangle for deom
                geom= box(lons[i] -0.3, lats[i]-0.3, lons[i]+0.3, lats[i]+0.3)
                geometries.append(geom)
            
            gdf = gdf.GoDataFrame(districts_data, geometry = geometries, crs="EPSG:4326")

            #save sample data
            output_file = self.config.GEO_DATA / "sample_rwanda_districts.geojson"
            gdf.to_file(output_file, driver= "GeoJSON")

            print(f"Sample boundary data saved to: {output_file}")
            return gdf
    
    def create_sample_vulnerability_data(self):
         """Create sample vunelability data"""
         print("Creating sample vulnerability data...")

         districts = ['Kigali', 'Musanze','Huye', 'Rwamagana','Rubavu']
         data={
            'district': districts,
            'population': [1745555,476522,381900,484953,546683],
            'poverty_rate': [0.15, 0.45, 0.38, 0.40, 0.35],
            'literacy_rate': [0.85, 0.65, 0.70, 0.60, 0.75],
            'hospital_distance_km': [2.5, 8.0, 8.0, 12.0, 9.0],
            'vulnerability_index': [0.3, 0.5, 0.7, 0.7, 0.5]
         }
         df= pd.DataFrame(data)

         #save sample data
         output_file = self.config.SOCIO_DATA / "sample_vulnerability.csv"
         df.to_csv(output_file, index = False)
         print(f"Sample vulnarability data saved to: {output_file}")

         return df
