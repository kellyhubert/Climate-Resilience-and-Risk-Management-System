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