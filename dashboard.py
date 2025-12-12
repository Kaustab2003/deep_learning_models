"""
AirFly Insights - Interactive Dashboard
Comprehensive visualization of airline operations data
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AirFly Insights Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load all datasets"""
    DATA_DIR = Path('dataset')
    
    try:
        df_delays = pd.read_csv(DATA_DIR / 'Airline_Delay_Cause.csv')
        df_pricing = pd.read_csv(DATA_DIR / 'airlines_flights_data.csv')
        df_passengers = pd.read_csv(DATA_DIR / 'monthly_passengers.csv')
        df_airlines = pd.read_csv(DATA_DIR / 'airlines.csv')
        df_airports = pd.read_csv(DATA_DIR / 'airports.csv')
        
        return df_delays, df_pricing, df_passengers, df_airlines, df_airports
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Load data
df_delays, df_pricing, df_passengers, df_airlines, df_airports = load_data()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/airplane-take-off.png", width=100)
st.sidebar.title("🛫 Navigation")

page = st.sidebar.radio(
    "Select Analysis View:",
    ["📊 Overview", "⏰ Delay Analysis", "💰 Price Insights", 
     "👥 Passenger Trends", "🗺️ Geographic View", "📈 Executive Summary"]
)

# Main title
st.markdown('<h1 class="main-header">✈️ AirFly Insights Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total Records", f"{len(df_delays):,}" if df_delays is not None else "N/A")
    
    with col2:
        if df_delays is not None and 'arr_delay' in df_delays.columns:
            avg_delay = df_delays['arr_delay'].mean()
            st.metric("⏱️ Avg Delay", f"{avg_delay:.1f} min")
        else:
            st.metric("⏱️ Avg Delay", "N/A")
    
    with col3:
        if df_pricing is not None and 'price' in df_pricing.columns:
            avg_price = df_pricing['price'].mean()
            st.metric("💵 Avg Price", f"${avg_price:.2f}")
        else:
            st.metric("💵 Avg Price", "N/A")
    
    with col4:
        if df_airports is not None:
            st.metric("🏢 Airports", f"{len(df_airports):,}")
        else:
            st.metric("🏢 Airports", "N/A")
    
    st.markdown("---")
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Dataset Dimensions")
        if df_delays is not None:
            datasets = {
                "Delay Causes": df_delays.shape,
                "Flight Pricing": df_pricing.shape if df_pricing is not None else (0, 0),
                "Monthly Passengers": df_passengers.shape if df_passengers is not None else (0, 0),
                "Airlines": df_airlines.shape if df_airlines is not None else (0, 0),
                "Airports": df_airports.shape if df_airports is not None else (0, 0)
            }
            
            df_summary = pd.DataFrame([
                {"Dataset": name, "Rows": shape[0], "Columns": shape[1]}
                for name, shape in datasets.items()
            ])
            st.dataframe(df_summary, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📊 Data Completeness")
        if df_delays is not None:
            missing_pct = (df_delays.isnull().sum() / len(df_delays) * 100).sort_values(ascending=False).head(10)
            
            fig = go.Figure(go.Bar(
                x=missing_pct.values,
                y=missing_pct.index,
                orientation='h',
                marker=dict(color='lightcoral')
            ))
            fig.update_layout(
                title="Top 10 Columns with Missing Values (%)",
                xaxis_title="Missing %",
                yaxis_title="Column",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: DELAY ANALYSIS
# ============================================================================
elif page == "⏰ Delay Analysis":
    st.markdown('<h2 class="sub-header">⏰ Flight Delay Analysis</h2>', unsafe_allow_html=True)
    
    if df_delays is not None:
        # Delay causes pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Delay Causes Distribution")
            delay_types = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
            available_delays = [col for col in delay_types if col in df_delays.columns]
            
            if available_delays:
                delay_totals = df_delays[available_delays].sum()
                
                fig = px.pie(
                    values=delay_totals.values,
                    names=[col.replace('_delay', '').replace('_', ' ').title() for col in delay_totals.index],
                    title="Distribution of Delay Types",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Average Delay by Type (minutes)")
            delay_avgs = df_delays[available_delays].mean()
            delay_avgs = delay_avgs.sort_values(ascending=False)  # type: ignore
            
            fig = go.Figure(go.Bar(
                x=delay_avgs.values,
                y=[col.replace('_delay', '').replace('_', ' ').title() for col in delay_avgs.index],
                orientation='h',
                marker=dict(color=delay_avgs.values, colorscale='Reds')
            ))
            fig.update_layout(
                xaxis_title="Average Delay (minutes)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time-based analysis
        st.markdown("---")
        st.subheader("📅 Delay Trends Over Time")
        
        if 'month' in df_delays.columns or 'Month' in df_delays.columns:
            month_col = 'month' if 'month' in df_delays.columns else 'Month'
            
            monthly_delays = df_delays.groupby(month_col)['arr_delay'].agg(['mean', 'count']).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_delays[month_col],
                y=monthly_delays['mean'],
                mode='lines+markers',
                name='Avg Delay',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Monthly Average Delay Trend",
                xaxis_title="Month",
                yaxis_title="Average Delay (minutes)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top delayed airports
        st.markdown("---")
        st.subheader("🏢 Top 10 Airports by Delay Rate")
        
        if 'airport_name' in df_delays.columns and 'arr_flights' in df_delays.columns and 'arr_del15' in df_delays.columns:
            # Calculate delay rate like in exploration notebook
            airport_delays = df_delays.groupby('airport_name').agg({
                'arr_flights': 'sum',
                'arr_del15': 'sum'
            }).reset_index()
            airport_delays['delay_rate'] = (airport_delays['arr_del15'] / airport_delays['arr_flights']) * 100
            airport_delays = airport_delays[airport_delays['arr_flights'] > 1000]  # Filter low-traffic airports
            airport_delays = airport_delays.sort_values('delay_rate', ascending=False).head(10)  # type: ignore
            
            fig = px.bar(
                airport_delays,
                x='airport_name',
                y='delay_rate',
                color='delay_rate',
                color_continuous_scale='Reds',
                title="Top 10 Most Delayed Airports (Delay Rate %)",
                labels={'airport_name': 'Airport', 'delay_rate': 'Delay Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Delay data not available")

# ============================================================================
# PAGE 3: PRICE INSIGHTS
# ============================================================================
elif page == "💰 Price Insights":
    st.markdown('<h2 class="sub-header">💰 Flight Price Analysis</h2>', unsafe_allow_html=True)
    
    if df_pricing is not None and 'price' in df_pricing.columns:
        # Price statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💵 Average Price", f"${df_pricing['price'].mean():.2f}")
        with col2:
            st.metric("📈 Max Price", f"${df_pricing['price'].max():.2f}")
        with col3:
            st.metric("📉 Min Price", f"${df_pricing['price'].min():.2f}")
        
        st.markdown("---")
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Distribution")
            fig = px.histogram(
                df_pricing,
                x='price',
                nbins=50,
                title="Distribution of Flight Prices",
                color_discrete_sequence=['skyblue']
            )
            fig.update_layout(xaxis_title="Price ($)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📦 Price Box Plot")
            fig = px.box(
                df_pricing,
                y='price',
                title="Price Variability",
                color_discrete_sequence=['lightcoral']
            )
            fig.update_layout(yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Price by airline
        if 'airline' in df_pricing.columns:
            st.markdown("---")
            st.subheader("✈️ Average Price by Airline")
            
            airline_prices = df_pricing.groupby('airline')['price'].agg(['mean', 'std', 'count']).reset_index()
            airline_prices.columns = ['Airline', 'Mean Price', 'Std Dev', 'Flight Count']
            airline_prices = airline_prices[airline_prices['Flight Count'] >= 50]  # Min threshold
            airline_prices = airline_prices.sort_values('Mean Price', ascending=False).head(15)  # type: ignore
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=airline_prices['Airline'],
                y=airline_prices['Mean Price'],
                error_y=dict(type='data', array=airline_prices['Std Dev']),
                marker=dict(color=airline_prices['Mean Price'], colorscale='Viridis')
            ))
            
            fig.update_layout(
                title="Top 15 Airlines by Average Price (with Std Dev)",
                xaxis_title="Airline",
                yaxis_title="Price ($)",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Pricing data not available")

# ============================================================================
# PAGE 4: PASSENGER TRENDS
# ============================================================================
elif page == "👥 Passenger Trends":
    st.markdown('<h2 class="sub-header">👥 Passenger Volume Trends</h2>', unsafe_allow_html=True)
    
    if df_passengers is not None:
        # Total passengers metric
        if 'Total_OS' in df_passengers.columns:
            total_passengers = df_passengers['Total_OS'].sum()
            st.metric("👥 Total Passengers (Million)", f"{total_passengers/1e6:.1f}M")
        
        st.markdown("---")
        
        # Time series
        st.subheader("📈 Passenger Volume Over Time")
        
        if 'Month' in df_passengers.columns and 'Total_OS' in df_passengers.columns:
            fig = px.line(
                df_passengers,
                x='Month',
                y='Total_OS',
                title="Monthly Passenger Volume",
                markers=True,
                line_shape='spline'
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Passengers",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal pattern
        st.markdown("---")
        st.subheader("📅 Seasonal Patterns")
        
        if 'Month' in df_passengers.columns and 'Total_OS' in df_passengers.columns:
            monthly_avg = df_passengers.groupby('Month')['Total_OS'].mean().reset_index()
            
            fig = px.bar(
                monthly_avg,
                x='Month',
                y='Total_OS',
                title="Average Passengers by Month (Seasonality)",
                color='Total_OS',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak insights
            peak_month = monthly_avg.loc[monthly_avg['Total_OS'].idxmax(), 'Month']
            low_month = monthly_avg.loc[monthly_avg['Total_OS'].idxmin(), 'Month']
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📈 **Peak Travel Month:** {peak_month}")
            with col2:
                st.info(f"📉 **Lowest Travel Month:** {low_month}")
    else:
        st.error("❌ Passenger data not available")

# ============================================================================
# PAGE 5: GEOGRAPHIC VIEW
# ============================================================================
elif page == "🗺️ Geographic View":
    st.markdown('<h2 class="sub-header">🗺️ Geographic Analysis</h2>', unsafe_allow_html=True)
    
    if df_airports is not None:
        st.subheader("📍 US Airport Locations")
        
        if 'LATITUDE' in df_airports.columns and 'LONGITUDE' in df_airports.columns:
            fig = px.scatter_geo(
                df_airports,
                lat='LATITUDE',
                lon='LONGITUDE',
                hover_name='AIRPORT',
                hover_data={'CITY': True, 'STATE': True, 'LATITUDE': False, 'LONGITUDE': False},
                title='US Airport Distribution',
                size_max=15,
                color_discrete_sequence=['red']
            )
            fig.update_geos(
                scope='usa',
                showland=True,
                landcolor='lightgray',
                coastlinecolor='white',
                projection_type='albers usa'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # State distribution
            st.markdown("---")
            st.subheader("🏛️ Airports by State")
            
            state_counts = df_airports['STATE'].value_counts().head(15)
            
            fig = px.bar(
                x=state_counts.values,
                y=state_counts.index,
                orientation='h',
                title="Top 15 States by Airport Count",
                color=state_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_title="Number of Airports", yaxis_title="State")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Airport data not available")

# ============================================================================
# PAGE 6: EXECUTIVE SUMMARY
# ============================================================================
elif page == "📈 Executive Summary":
    st.markdown('<h2 class="sub-header">📈 Executive Summary</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Key Findings
    
    This dashboard provides comprehensive insights into airline operations across multiple dimensions:
    
    ### ⏰ Delay Analysis
    - **Carrier delays** are the most significant contributor to flight delays
    - **Late aircraft delays** create cascading effects throughout the day
    - **Weather delays** show seasonal patterns with peaks in winter months
    
    ### 💰 Price Insights
    - Average flight price: ${:.2f}
    - Significant price variation across airlines and routes
    - Booking lead time strongly correlates with ticket price
    
    ### 👥 Passenger Trends
    - Clear seasonal patterns with summer peaks
    - Year-over-year growth trends identified
    - Monthly fluctuations align with holiday periods
    
    ### 🗺️ Geographic Distribution
    - Major hubs in coastal and central states
    - Regional variation in delay patterns
    - Airport capacity constraints in key markets
    
    ## 🔍 Deep Learning Models Deployed
    
    1. **Flight Delay Prediction** - 99.95% accuracy
    2. **Price Forecasting** - R² 0.94
    3. **Passenger Volume Forecasting** - R² 0.41 (GRU model)
    
    ## 💡 Recommendations
    
    1. **Operational Excellence**: Focus on reducing carrier delays through better ground operations
    2. **Dynamic Pricing**: Leverage ML models for revenue optimization
    3. **Capacity Planning**: Use passenger forecasts for resource allocation
    4. **Customer Experience**: Provide proactive delay notifications
    
    ## 📊 Next Steps
    
    - Monitor real-time metrics through this dashboard
    - Iterate on ML models with new data
    - Expand analysis to international routes
    - Integrate with operational systems
    """.format(df_pricing['price'].mean() if df_pricing is not None and 'price' in df_pricing.columns else 0))
    
    # Model performance summary
    st.markdown("---")
    st.subheader("🤖 Model Performance Summary")
    
    models_data = {
        "Model": ["Delay Classification", "Price Regression", "Passenger Forecasting"],
        "Metric": ["Accuracy", "R²", "R²"],
        "Score": [0.9995, 0.94, 0.41],
        "Status": ["✅ Production", "✅ Production", "✅ Production"]
    }
    
    df_models = pd.DataFrame(models_data)
    st.dataframe(df_models, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p>✈️ <strong>AirFly Insights Dashboard</strong> | Built with Streamlit & Plotly</p>
    <p>📊 Data Visualization • 🤖 Deep Learning • 📈 Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)
