# Aviation Deep Learning Projects 🛫

A comprehensive collection of deep learning projects for aviation data analysis, including flight delay prediction, ticket price forecasting, and passenger volume time series modeling.

## 📊 Project Overview

This repository contains 5 end-to-end Jupyter notebooks implementing deep learning solutions for real-world aviation datasets:

1. **Data Exploration & Analysis** - Comprehensive EDA with visualizations
2. **Data Preprocessing & Feature Engineering** - Clean, transform, and prepare data
3. **Flight Delay Prediction** - Binary/multi-class classification and regression
4. **Airline Price Prediction** - Deep neural network for ticket pricing
5. **Passenger Forecasting** - LSTM time series for volume prediction

## 📁 Project Structure

```
deep_learning_models/
│
├── dataset/                                    # Raw data files (CSV)
│   ├── Airline_Delay_Cause.csv               # Delay statistics (171K rows)
│   ├── airlines_flights_data.csv             # Indian flight pricing (300K rows)
│   ├── monthly_passengers.csv                # Global passenger data (7K rows)
│   ├── airports.csv                          # Airport metadata (324 airports)
│   ├── airlines.csv                          # Airline lookup table
│   ├── global_holidays.csv                   # Holiday calendar (44K rows)
│   └── GlobalWeatherRepository.csv           # Weather data (108K rows)
│
├── processed_data/                            # Preprocessed datasets (created by notebook 02)
│   ├── delay_*.npy                           # Delay prediction arrays
│   ├── price_*.npy                           # Price prediction arrays
│   ├── passenger_*.npy                       # Passenger forecasting arrays
│   └── *.pkl                                 # Encoders and scalers
│
├── models/                                    # Trained models (created during training)
│   ├── delay_binary_model.pth                # Binary delay classifier
│   ├── delay_multiclass_model.pth            # Multi-class delay classifier
│   ├── delay_regression_model.pth            # Delay duration regressor
│   ├── price_prediction_model.pth            # Price forecasting model
│   └── passenger_lstm_model.pth              # LSTM passenger forecaster
│
├── 01_data_exploration.ipynb                  # EDA and visualization
├── 02_data_preprocessing.ipynb                # Feature engineering
├── 03_delay_prediction_basic.ipynb            # Flight delay models
├── 04_price_prediction_dnn.ipynb              # Ticket price models
├── 05_passenger_forecasting_lstm.ipynb        # Time series LSTM
│
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- 8GB+ RAM

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**:
```powershell
jupyter notebook
```

5. **Run notebooks in order**:
   - Start with `01_data_exploration.ipynb`
   - Follow the sequence through `05_passenger_forecasting_lstm.ipynb`

## 📚 Notebook Descriptions

### 01 - Data Exploration & Analysis
**Goal:** Understand the aviation datasets  
**Key Features:**
- Load and inspect 7 different datasets
- Statistical summaries and missing value analysis
- Visualizations: delay trends, price distributions, passenger seasonality
- Airport geographic mapping
- Weather correlation analysis

**Outputs:** Insights and visualizations for understanding data quality

---

### 02 - Data Preprocessing & Feature Engineering
**Goal:** Prepare clean, model-ready datasets  
**Key Features:**
- Handle missing values and outliers
- Feature engineering: temporal encodings, lag features, rolling statistics
- Categorical encoding (LabelEncoder for airlines, airports, routes)
- Normalization and scaling (StandardScaler)
- Time-based train/validation/test splits

**Outputs:** 
- Preprocessed numpy arrays saved to `processed_data/`
- Encoders and scalers saved as pickle files

---

### 03 - Flight Delay Prediction (Basic Neural Networks)
**Goal:** Build baseline deep learning models for flight delays  
**Key Features:**
- **Binary Classification:** Delayed (15+ min) vs On-time
- **Multi-class Classification:** Predict delay cause (carrier, weather, NAS, security, late aircraft)
- **Regression:** Predict delay duration in minutes
- Feedforward neural networks with batch normalization and dropout
- Training with early stopping and learning rate scheduling

**Results:**
- Binary accuracy: ~XX% (depends on your data)
- Multi-class accuracy: ~XX%
- Regression MAE: ~XX minutes

**Outputs:** Trained models saved to `models/`

---

### 04 - Airline Price Prediction (Deep Neural Network)
**Goal:** Forecast ticket prices using deep learning  
**Key Features:**
- Deep neural network with 4 hidden layers
- Log-transformed target for better regression performance
- Feature interactions learned through deep architecture
- Comprehensive evaluation metrics (MAE, RMSE, MAPE, R²)
- Visualizations: actual vs predicted, residuals, error distributions

**Results:**
- MAE: ~₹X,XXX
- MAPE: ~X.X%
- R²: ~0.XX

**Outputs:** Trained price prediction model

---

### 05 - Passenger Forecasting (LSTM Time Series)
**Goal:** Forecast monthly passenger volumes using LSTM  
**Key Features:**
- LSTM architecture with 2 layers for temporal dependencies
- Sequence-to-one prediction (12-month lookback)
- Multi-step ahead forecasting capability
- Gradient clipping to prevent exploding gradients
- Time series visualizations and error analysis

**Results:**
- MAE: ~X.XX million passengers
- MAPE: ~XX%
- R²: ~0.XX

**Outputs:** Trained LSTM model

## 🎯 Key Learning Objectives

### Beginner Level
✅ Load and explore datasets with pandas  
✅ Create visualizations with matplotlib/seaborn  
✅ Understand data preprocessing and feature engineering  
✅ Build basic neural networks with PyTorch  

### Intermediate Level
✅ Implement classification and regression models  
✅ Handle imbalanced datasets  
✅ Use train/validation/test splits properly  
✅ Apply regularization techniques (dropout, weight decay)  
✅ Interpret model metrics and visualizations  

### Advanced Level
✅ Build LSTM networks for time series forecasting  
✅ Implement custom PyTorch datasets and dataloaders  
✅ Use learning rate scheduling and early stopping  
✅ Perform multi-step ahead forecasting  
✅ Analyze model errors and residuals  

## 📊 Dataset Information

| Dataset | Rows | Description | Use Case |
|---------|------|-------------|----------|
| Airline_Delay_Cause.csv | 171,668 | Monthly delay statistics (2013-2023) | Delay prediction |
| airlines_flights_data.csv | 300,155 | Indian domestic flight pricing | Price forecasting |
| monthly_passengers.csv | 7,244 | Global monthly passenger volumes (2010-2017) | Time series forecasting |
| airports.csv | 324 | US airport metadata with coordinates | Geographic features |
| global_holidays.csv | 44,395 | International holiday calendar | Temporal features |
| GlobalWeatherRepository.csv | 107,965 | Weather data for 195+ countries | Environmental features |

## 🧠 Model Architectures

### Delay Prediction (Feedforward NN)
```
Input (16 features) → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
                    → Output
```

### Price Prediction (Deep NN)
```
Input (15 features) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(64)  → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(32)  → BatchNorm → ReLU → Dropout(0.3)
                    → Dense(1)   # Price output
```

### Passenger Forecasting (LSTM)
```
Input (seq_len=12, features=14) → LSTM(128, 2 layers) → Dense(64) → ReLU
                                                       → Dense(32) → ReLU
                                                       → Dense(1)  # Volume output
```

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch 2.0+
- **Data Processing:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn, plotly
- **Development:** Jupyter Notebook

## 📈 Performance Tips

1. **GPU Acceleration:** If you have a CUDA-capable GPU, PyTorch will automatically use it
2. **Batch Size:** Adjust based on your RAM (512 for delay/price, 64 for LSTM)
3. **Early Stopping:** Models use early stopping to prevent overfitting
4. **Learning Rate:** Uses ReduceLROnPlateau for automatic adjustment

## 🔧 Troubleshooting

**Issue:** Out of memory errors  
**Solution:** Reduce batch size in notebook cells

**Issue:** Slow training  
**Solution:** 
- Reduce number of epochs
- Use smaller hidden dimensions
- Enable GPU if available

**Issue:** Poor model performance  
**Solution:**
- Check data preprocessing in notebook 02
- Experiment with hyperparameters
- Add more features or try different architectures

## 📝 Next Steps & Improvements

### Advanced Architectures
- [ ] Implement TabNet for interpretable deep learning
- [ ] Try Temporal Fusion Transformer for multi-horizon forecasting
- [ ] Build Graph Neural Networks for airport network analysis
- [ ] Ensemble multiple models for better predictions

### Feature Engineering
- [ ] Add weather-flight temporal alignment
- [ ] Create airline-route interaction features
- [ ] Incorporate holiday impact analysis
- [ ] Use geospatial distance calculations

### Deployment
- [ ] Create REST API with Flask/FastAPI
- [ ] Build interactive dashboard with Streamlit
- [ ] Containerize with Docker
- [ ] Deploy to cloud (AWS/Azure/GCP)

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with hyperparameters
- Try different architectures
- Add new features
- Improve visualizations

## 📄 License

This project is for educational purposes. Datasets are assumed to be publicly available or used under fair use for learning.

## 🙏 Acknowledgments

- Aviation datasets from various public sources
- PyTorch documentation and tutorials
- scikit-learn for preprocessing utilities

---

## 🎓 Learning Path

**Week 1-2:** Data Exploration & Preprocessing  
→ Run notebooks 01 and 02, understand the data

**Week 3-4:** Basic Deep Learning  
→ Complete notebook 03, learn classification/regression

**Week 5-6:** Advanced Regression  
→ Work through notebook 04, master price prediction

**Week 7-8:** Time Series with LSTM  
→ Finish notebook 05, understand sequence modeling

**Week 9+:** Experimentation & Improvement  
→ Try your own ideas, improve models, deploy solutions

---

**Happy Learning! 🚀✈️**

For questions or issues, review notebook comments and PyTorch documentation.
