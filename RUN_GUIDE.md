# Quick Start Guide - Optimized Models

## 🚀 Run All Notebooks

Execute these commands in PowerShell:

```powershell
# 1. Activate virtual environment
cd "C:\Users\Kaustab das\Desktop\deep_learning_models"
.\venv\Scripts\Activate.ps1

# 2. Launch Jupyter
jupyter notebook
```

## 📋 Execution Order

### Step 1: Data Exploration (Optional - for visualization)
- Open `01_data_exploration.ipynb`
- Run All Cells (Kernel → Restart & Run All)
- Expected time: ~5 minutes

### Step 2: Data Preprocessing (REQUIRED)
- Open `02_data_preprocessing.ipynb`
- Run All Cells (Kernel → Restart & Run All)
- Creates `processed_data/` folder with:
  - `delay_*.npy`
  - `price_*.npy`
  - `passenger_*.npy`
  - `*.pkl` (encoders)
- Expected time: ~3-5 minutes

### Step 3: Delay Prediction (Already trained)
- `03_delay_prediction_basic.ipynb` 
- ✅ Already excellent (99.88% accuracy)
- No need to re-run unless you want

### Step 4: Price Prediction (OPTIMIZED) ⭐
- Open `04_price_prediction_dnn.ipynb`
- **Important**: Restart kernel before running
- Run All Cells
- **Improvements**:
  - Larger model (512→256→128→64)
  - AdamW optimizer
  - Gradient clipping
  - Better regularization
- Expected time: ~15-25 minutes
- Expected improvement: MAPE 60% → <40%, R² 0.33 → >0.60

### Step 5: Passenger LSTM (OPTIMIZED) ⭐
- Open `05_passenger_forecasting_lstm.ipynb`
- **Important**: Restart kernel before running
- Run All Cells
- **Improvements**:
  - Bidirectional LSTM
  - Attention mechanism
  - HuberLoss (robust to outliers)
  - OneCycleLR scheduler
  - Layer normalization
- Expected time: ~20-35 minutes
- Expected improvement: MAPE 86% → <30%, R² -0.17 → >0.70

## ⚡ Quick Commands

### In Jupyter Notebook:
- **Restart Kernel**: `Kernel` → `Restart Kernel...`
- **Run All**: `Kernel` → `Restart & Run All`
- **Run Single Cell**: `Shift + Enter`
- **Stop Execution**: Press `⏹` (Stop button) or `Kernel` → `Interrupt`

## 📊 Expected Results After Optimization

| Model | Before | After (Target) |
|-------|--------|----------------|
| **Price MAE** | ₹12,314 | < ₹8,000 |
| **Price MAPE** | 59.9% | < 40% |
| **Price R²** | 0.33 | > 0.60 |
| **LSTM MAE** | 14,393M | < 5,000M |
| **LSTM MAPE** | 86.1% | < 30% |
| **LSTM R²** | -0.17 | > 0.70 |

## 🔍 Monitoring Training

### Watch for these indicators:

**Good Signs:**
- ✅ Train loss steadily decreasing
- ✅ Val loss decreasing (with small gap from train loss)
- ✅ No NaN values
- ✅ Gradients not exploding

**Warning Signs:**
- ⚠️ Val loss >> Train loss (overfitting)
- ⚠️ Loss = NaN (exploding gradients)
- ⚠️ No improvement after 20 epochs (learning rate too low)

## 🐛 Troubleshooting

### Problem: "processed_data folder not found"
**Solution**: Run notebook 02 first

### Problem: "CUDA out of memory"
**Solution**: 
- Reduce batch_size in the notebook
- Close other applications
- Restart kernel

### Problem: "Loss is NaN"
**Solution**:
- Already fixed with gradient clipping
- If still occurs, reduce learning rate by 50%

### Problem: "Training too slow"
**Solution**:
- Reduce `num_epochs` from 100 to 50
- Reduce model size (hidden_dims)
- Use CPU instead (set `device = 'cpu'`)

## 📝 Notes

- Both optimized models use **gradient clipping** to prevent exploding gradients
- **Notebook 04** uses AdamW with ReduceLROnPlateau
- **Notebook 05** uses AdamW with OneCycleLR (super-convergence)
- All improvements are already applied to the code
- Just need to **restart kernel and run all cells**

## 📧 After Training

Check the final output cells to see:
- Model performance metrics
- Saved model paths
- Comparison with baseline
- Next steps recommendations

Good luck! 🎯
