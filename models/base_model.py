"""
Base model class for risk assessment models."""

import pickle
import joblib
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

