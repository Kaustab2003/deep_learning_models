# 🚀 Quick Start Guide - AirFly Insights Dashboard

## Prerequisites
- Python 3.8+
- All datasets in `dataset/` folder

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Kaustab2003/deep_learning_models.git
cd deep_learning_models
```

2. **Create virtual environment:**
```bash
python -m venv venv
```

3. **Activate environment:**
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Dashboard

### Option 1: Streamlit Dashboard (Interactive)
```bash
streamlit run dashboard.py
```
Then open your browser to: **http://localhost:8501**

### Option 2: Jupyter Notebooks (Step-by-step)
```bash
jupyter notebook
```
Run notebooks in order:
1. `01_data_exploration.ipynb` - Visualizations
2. `02_data_preprocessing.ipynb` - Data prep
3. `03_delay_prediction_basic.ipynb` - Delay models
4. `04_price_prediction_dnn.ipynb` - Price models
5. `05_passenger_forecasting_lstm.ipynb` - Passenger forecasting

## Dashboard Navigation

### 📊 Overview Page
- Dataset dimensions and completeness
- Missing value analysis
- Quick statistics

### ⏰ Delay Analysis Page
- Delay causes breakdown (pie chart)
- Monthly delay trends
- Top 10 delayed airports
- Average delay by type

### 💰 Price Insights Page
- Price distribution histogram
- Price box plot
- Average price by airline with error bars
- Price statistics

### 👥 Passenger Trends Page
- Total passenger volume over time
- Seasonal patterns by month
- Peak/low travel months
- Growth trends

### 🗺️ Geographic View Page
- Interactive US airport map
- Airports by state (top 15)
- Geographic coverage statistics

### 📈 Executive Summary Page
- Key findings across all analyses
- ML model performance summary
- Strategic recommendations
- Next steps

## Troubleshooting

### Dashboard won't start
```bash
# Reinstall Streamlit
pip install --upgrade streamlit
```

### Data not loading
- Ensure all CSV files are in `dataset/` folder
- Check file names match exactly (case-sensitive)

### Port already in use
```bash
# Use different port
streamlit run dashboard.py --server.port 8502
```

### Missing dependencies
```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

## Key Features

✅ **Real-time Filtering** - Interactive controls on all visualizations  
✅ **Export Ready** - High-quality charts for presentations  
✅ **Mobile Responsive** - Works on all devices  
✅ **Fast Loading** - Caching for optimal performance  
✅ **6 Analysis Views** - Comprehensive coverage of all insights  

## Advanced Usage

### Custom Port
```bash
streamlit run dashboard.py --server.port 8080
```

### Network Access (Share with team)
```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

### Dark Mode
Click ⚙️ Settings → Theme → Dark

### Export Charts
Right-click any Plotly chart → Download plot as PNG

## Performance Tips

- Dashboard caches data on first load
- Subsequent page changes are instant
- For large datasets, consider sampling in code
- Clear cache: Click "⋮" menu → Clear cache

## Documentation

- **Executive Summary:** See `EXECUTIVE_SUMMARY.md`
- **Technical Details:** See `README.md`
- **Notebook Guide:** See `RUN_GUIDE.md`

## Support

**Issues:** https://github.com/Kaustab2003/deep_learning_models/issues  
**Repository:** https://github.com/Kaustab2003/deep_learning_models

---

🎉 **You're all set! Enjoy exploring AirFly Insights!** 🎉
