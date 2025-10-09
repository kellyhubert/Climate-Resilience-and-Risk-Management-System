"""
Train all Rwanda climate risk models
Run this BEFORE starting the dashboard
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.base_model import BaseRiskModel
from models.landslide_model import RwandaLandslideModel
from models.flood_model import RwandaFloodModel
from models.drought_model import RwandaDroughtModel

def main():
    print("=" * 70)
    print("TRAINING RWANDA CLIMATE RISK MODELS")
    print("=" * 70)
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / 'models' / 'trained'
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Models will be saved to: {models_dir}\n")
    
    # 1. Train Landslide Model
    print("=" * 70)
    print("1. TRAINING LANDSLIDE MODEL")
    print("=" * 70)
    try:
        landslide = RwandaLandslideModel()
        landslide.train()
        landslide.save_model(models_dir / 'rwanda_landslide_model.pkl')
        print("\n✅ Landslide model saved!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    # 2. Train Flood Model
    print("\n" + "=" * 70)
    print("2. TRAINING FLOOD MODEL")
    print("=" * 70)
    try:
        flood = RwandaFloodModel()
        flood.train()
        flood.save_model(models_dir / 'rwanda_flood_model.pkl')
        print("\n✅ Flood model saved!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    # 3. Train Drought Model
    print("\n" + "=" * 70)
    print("3. TRAINING DROUGHT MODEL")
    print("=" * 70)
    try:
        drought = RwandaDroughtModel()
        drought.train()
        drought.save_model(models_dir / 'rwanda_drought_model.pkl')
        print("\n✅ Drought model saved!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 70)
    print("\nYou can now start the dashboard:")
    print("  cd dashboard")
    print("  python app.py")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed. Please check the errors above.")
        sys.exit(1)