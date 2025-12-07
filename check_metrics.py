import torch
from pathlib import Path

MODELS_DIR = Path('models')

def check_model_metrics(model_name, file_name):
    path = MODELS_DIR / file_name
    if not path.exists():
        print(f"❌ {model_name}: Model file not found ({file_name})")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'test_metrics' in checkpoint:
            metrics = checkpoint['test_metrics']
            print(f"✅ {model_name} Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        else:
            print(f"⚠️ {model_name}: No metrics found in checkpoint dictionary.")
    except Exception as e:
        print(f"❌ {model_name}: Error loading checkpoint - {e}")

print("Checking saved model metrics...\n")
check_model_metrics("Price Prediction (NB 04)", "price_prediction_model.pth")
check_model_metrics("Passenger Forecasting (NB 05)", "passenger_lstm_model.pth")
