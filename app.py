import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Oil & Gas Well Data Analysis Platform",
    page_icon="🛢️",
    layout="wide"
)

# Title and description
st.title("🛢️ Oil & Gas Well Data Analysis Platform")
st.markdown("Upload and analyze oil and gas well production data with interactive visualization")

# Initialize session state
if 'production_data' not in st.session_state:
    st.session_state.production_data = None
if 'header_data' not in st.session_state:
    st.session_state.header_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_iqr_custom(df, column, sensitivity=1.5):
    """Detect outliers using IQR method with configurable sensitivity"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - sensitivity * IQR
    upper_bound = Q3 + sensitivity * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Decline curve modeling functions
def arps_exponential(t, qi, D):
    """Arps exponential decline model"""
    return qi * np.exp(-D * t)

def arps_harmonic(t, qi, D):
    """Arps harmonic decline model"""
    return qi / (1 + D * t)

def arps_hyperbolic(t, qi, D, b):
    """Arps hyperbolic decline model"""
    return qi / ((1 + b * D * t) ** (1 / b))

def compute_eur(model, qi, D, b=None, t_end=None):
    """Compute Estimated Ultimate Recovery (EUR)"""
    if model == 'exponential':
        return qi / D * (1 - np.exp(-D * t_end))
    elif model == 'harmonic':
        return qi * np.log(1 + D * t_end) / D
    elif model == 'hyperbolic':
        if b == 1:
            return qi / D * (1 - np.exp(-D * t_end))
        base = 1 + b * D * t_end
        if base <= 0:
            return np.nan
        return (qi / ((1 - b) * D)) * (1 - base ** (1 - 1 / b))

def fit_decline_models(production_df, r2_threshold=0.8):
    """Fit decline curve models and return best models for each well with R² filtering"""
    results = []
    
    # Determine which ID column to use
    id_column = 'WellName' if 'WellName' in production_df.columns else 'API_UWI'
    
    for well_id, group in production_df.groupby(id_column):
        group = group.sort_values('ProducingMonth')
        if len(group) < 6:
            continue
            
        # Calculate time in months
        group['t_months'] = (group['ProducingMonth'] - group['ProducingMonth'].min()).dt.days / 30.0
        t = group['t_months'].values
        q = group['Prod_BOE'].values
        t_end = t.max()
        
        if np.any(q <= 0) or np.all(q == 0) or np.isnan(q).any():
            continue
            
        models_to_try = [
            ('exponential', arps_exponential, [q[0], 0.01], ([0.01, 1e-5], [1e6, 1])),
            ('harmonic', arps_harmonic, [q[0], 0.01], ([0.01, 1e-5], [1e6, 1])),
            ('hyperbolic', arps_hyperbolic, [q[0], 0.01, 0.5], ([0.01, 1e-5, 0.01], [1e6, 1, 2]))
        ]
        
        best_model = None
        best_r2 = -np.inf
        
        for model_name, model_func, p0, bounds in models_to_try:
            try:
                popt, _ = curve_fit(model_func, t, q, p0=p0, bounds=bounds, maxfev=20000)
                q_pred = model_func(t, *popt)
                r2 = r2_score(q, q_pred)
                mae = mean_absolute_error(q, q_pred)
                rmse = np.sqrt(mean_squared_error(q, q_pred))
                
                # Calculate EUR
                if model_name == 'hyperbolic':
                    qi, D, b = popt
                    eur = compute_eur(model_name, qi, D, b=b, t_end=t_end)
                else:
                    qi, D = popt
                    b = np.nan
                    eur = compute_eur(model_name, qi, D, t_end=t_end)
                
                # Only consider models that meet R² threshold
                if r2 > best_r2 and eur > 0 and r2 >= r2_threshold:
                    best_r2 = r2
                    
                    # Get well name and API for display
                    well_name = well_id
                    api_uwi = group['API_UWI'].iloc[0] if 'API_UWI' in group.columns else well_id
                    
                    best_model = {
                        'WellName': well_name,
                        'API_UWI': api_uwi,
                        'well_id': well_id,  # Keep original ID for grouping
                        'model': model_name,
                        'qi': qi,
                        'D': D,
                        'b': b,
                        'EUR_BOE': eur,
                        'R2': r2,
                        'MAE': mae,
                        'RMSE': rmse,
                        't_data': t,
                        'q_data': q,
                        'q_pred': q_pred
                    }
            except:
                continue
                
        if best_model:
            results.append(best_model)
    
    return pd.DataFrame(results)

def predict_future_production(model_params, months_ahead=3):
    """Predict future production for given months"""
    model = model_params['model']
    qi = model_params['qi']
    D = model_params['D']
    b = model_params['b']
    
    # Get current max time
    t_current = model_params['t_data'].max()
    t_future = np.arange(t_current + 1, t_current + months_ahead + 1)
    
    if model == 'exponential':
        q_future = arps_exponential(t_future, qi, D)
    elif model == 'harmonic':
        q_future = arps_harmonic(t_future, qi, D)
    elif model == 'hyperbolic':
        q_future = arps_hyperbolic(t_future, qi, D, b)
    
    return t_future, q_future

def calculate_advanced_metrics(actual, predicted):
    """Calculate advanced error metrics including MAPE"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8))) * 100
    
    # Bias (systematic over/under prediction)
    bias = np.mean(predicted - actual)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Bias': bias
    }

def calculate_residuals(actual, predicted):
    """Calculate residuals and their statistics"""
    residuals = actual - predicted
    
    # Normalized residuals
    std_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
    
    return {
        'residuals': residuals,
        'std_residuals': std_residuals,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }

def calculate_cumulative_eur_error(model_params):
    """Calculate EUR prediction accuracy"""
    actual_eur = np.sum(model_params['q_data'])  # Simple cumulative from data
    predicted_eur = model_params['EUR_BOE']
    
    eur_error_pct = ((predicted_eur - actual_eur) / actual_eur) * 100 if actual_eur > 0 else 0
    
    return {
        'actual_eur': actual_eur,
        'predicted_eur': predicted_eur,
        'eur_error_pct': eur_error_pct
    }

def process_production_data(df):
    """Process production data with cleaning and feature engineering"""
    df_processed = df.copy()
    
    # Remove completely empty columns
    df_processed = df_processed.dropna(axis=1, how='all')
    
    # Define production columns for outlier detection
    production_cols = [
        'Prod_BOE', 'Prod_MCFE', 'GasProd_MCF', 'LiquidsProd_BBL',
        'WaterProd_BBL', 'RepGasProd_MCF', 'CDProd_BOEPerDAY',
        'CDProd_MCFEPerDAY', 'CDLiquids_BBLPerDAY', 'CDGas_MCFPerDAY',
        'CDWater_BBLPerDAY', 'CDRepGas_MCFPerDAY', 'PDProd_BOEPerDAY',
        'PDProd_MCFEPerDAY', 'PDLiquids_BBLPerDAY', 'PDGas_MCFPerDAY',
        'PDWater_BBLPerDAY', 'PDRepGas_MCFPerDAY', 'CumProd_BOE',
        'CumProd_MCFE', 'CumLiquids_BBL', 'CumGas_MCF', 'CumWater_BBL',
        'CumRepGas_MCF'
    ]
    
    # Filter columns that actually exist in the dataframe
    existing_prod_cols = [col for col in production_cols if col in df_processed.columns]
    
    # Check for zero columns and replace with NaN, then fill with median
    zero_columns = df_processed.columns[(df_processed == 0).any()]
    
    if len(zero_columns) > 0:
        df_processed[zero_columns] = df_processed[zero_columns].replace(0, np.nan)
        
        for col in zero_columns:
            if col in existing_prod_cols:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
    
    # Convert ProducingMonth to datetime
    if 'ProducingMonth' in df_processed.columns:
        df_processed['ProducingMonth'] = pd.to_datetime(
            df_processed['ProducingMonth'], 
            format='mixed', 
            errors='coerce', 
            dayfirst=False
        )
        
        # Calculate months since start for each well
        if 'API_UWI' in df_processed.columns:
            df_processed['FirstProdMonth'] = df_processed.groupby('API_UWI')['ProducingMonth'].transform('min')
            df_processed['MonthsSinceStart'] = (
                (df_processed['ProducingMonth'].dt.year - df_processed['FirstProdMonth'].dt.year) * 12 +
                (df_processed['ProducingMonth'].dt.month - df_processed['FirstProdMonth'].dt.month)
            )
            
            # Sort by API_UWI and ProducingMonth for cumulative calculations
            df_processed = df_processed.sort_values(['API_UWI', 'ProducingMonth'])
            
            # Calculate cumulative production metrics
            if 'Prod_BOE' in df_processed.columns:
                df_processed['CumProd_BOE'] = df_processed.groupby('API_UWI')['Prod_BOE'].cumsum()
            if 'GasProd_MCF' in df_processed.columns:
                df_processed['CumGas_MCF'] = df_processed.groupby('API_UWI')['GasProd_MCF'].cumsum()
            if 'WaterProd_BBL' in df_processed.columns:
                df_processed['CumWater_BBL'] = df_processed.groupby('API_UWI')['WaterProd_BBL'].cumsum()
            if 'Prod_MCFE' in df_processed.columns:
                df_processed['CumProd_MCFE'] = df_processed.groupby('API_UWI')['Prod_MCFE'].cumsum()
            if 'LiquidsProd_BBL' in df_processed.columns:
                df_processed['CumLiquidsProd_BBL'] = df_processed.groupby('API_UWI')['LiquidsProd_BBL'].cumsum()
            if 'RepGasProd_MCF' in df_processed.columns:
                df_processed['CumRepGasProd_MCF'] = df_processed.groupby('API_UWI')['RepGasProd_MCF'].cumsum()
    
    return df_processed, existing_prod_cols

def analyze_well_quality(production_df, header_df=None, outlier_sensitivity=1.5):
    """Analyze well data quality and create labels with configurable outlier sensitivity"""
    well_analysis = {}
    
    if 'API_UWI' not in production_df.columns:
        return well_analysis
    
    # Get unique wells
    wells = production_df['API_UWI'].unique()
    
    # Define production columns for outlier detection
    production_cols = [
        'Prod_BOE', 'Prod_MCFE', 'GasProd_MCF', 'LiquidsProd_BBL',
        'WaterProd_BBL', 'RepGasProd_MCF'
    ]
    existing_prod_cols = [col for col in production_cols if col in production_df.columns]
    
    for well in wells:
        well_data = production_df[production_df['API_UWI'] == well]
        
        # Initialize well analysis
        analysis = {
            'well_id': well,
            'has_outliers': False,
            'has_null_values': False,
            'insufficient_data': False,
            'total_months': 0,
            'outlier_columns': [],
            'null_columns': [],
            'outlier_details': [],
            'status': 'Normal'
        }
        
        # Check for insufficient data (less than 6 months)
        if 'MonthsSinceStart' in well_data.columns:
            max_months = well_data['MonthsSinceStart'].max()
            analysis['total_months'] = max_months if pd.notna(max_months) else 0
            if analysis['total_months'] < 6:
                analysis['insufficient_data'] = True
        else:
            analysis['total_months'] = len(well_data)
            if analysis['total_months'] < 6:
                analysis['insufficient_data'] = True
        
        # Check for null values
        for col in existing_prod_cols:
            if well_data[col].isnull().any():
                analysis['has_null_values'] = True
                analysis['null_columns'].append(col)
        
        # Check for outliers with configurable sensitivity
        for col in existing_prod_cols:
            if len(well_data) > 0 and well_data[col].notna().sum() > 0:
                try:
                    Q1 = well_data[col].quantile(0.25)
                    Q3 = well_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - outlier_sensitivity * IQR
                    upper_bound = Q3 + outlier_sensitivity * IQR
                    
                    outliers = well_data[(well_data[col] < lower_bound) | (well_data[col] > upper_bound)]
                    if len(outliers) > 0:
                        analysis['has_outliers'] = True
                        analysis['outlier_columns'].append(col)
                        analysis['outlier_details'].append(f"Outlier-{col}")
                except:
                    continue
        
        # Determine status with specific outlier column information
        statuses = []
        if analysis['has_outliers']:
            # Create detailed outlier status
            outlier_status = ', '.join(analysis['outlier_details'])
            statuses.append(outlier_status)
        if analysis['has_null_values']:
            statuses.append('Null Values')
        if analysis['insufficient_data']:
            statuses.append('Insufficient Data')
        
        if statuses:
            analysis['status'] = ', '.join(statuses)
        
        well_analysis[well] = analysis
    
    return well_analysis

def create_map_visualization(header_df, well_analysis):
    """Create interactive map with well locations and quality indicators"""
    if header_df is None or len(header_df) == 0:
        return None
    
    # Check for required coordinate columns
    lat_cols = [col for col in header_df.columns if 'lat' in col.lower()]
    lon_cols = [col for col in header_df.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No latitude/longitude columns found in header data")
        return None
    
    # Use first available lat/lon columns
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Prepare map data
    map_data = header_df.copy()
    
    # Add well analysis data
    if 'API_UWI' in map_data.columns:
        map_data['Status'] = map_data['API_UWI'].map(
            lambda x: well_analysis.get(x, {}).get('status', 'Normal')
        )
        map_data['Total_Months'] = map_data['API_UWI'].map(
            lambda x: well_analysis.get(x, {}).get('total_months', 0)
        )
        map_data['Has_Outliers'] = map_data['API_UWI'].map(
            lambda x: well_analysis.get(x, {}).get('has_outliers', False)
        )
        map_data['Has_Null_Values'] = map_data['API_UWI'].map(
            lambda x: well_analysis.get(x, {}).get('has_null_values', False)
        )
        map_data['Insufficient_Data'] = map_data['API_UWI'].map(
            lambda x: well_analysis.get(x, {}).get('insufficient_data', False)
        )
    else:
        map_data['Status'] = 'Unknown'
        map_data['Total_Months'] = 0
        map_data['Has_Outliers'] = False
        map_data['Has_Null_Values'] = False
        map_data['Insufficient_Data'] = False
    
    # Remove rows with invalid coordinates
    map_data = map_data.dropna(subset=[lat_col, lon_col])
    
    if len(map_data) == 0:
        st.error("No valid coordinate data found")
        return None
    
    # Create enhanced color mapping for specific outlier columns
    def get_status_color(status):
        if pd.isna(status) or status == 'Normal':
            return 'green'
        elif 'Outlier-' in str(status):
            return 'red'
        elif 'Null Values' in str(status):
            return 'orange'
        elif 'Insufficient Data' in str(status):
            return 'blue'
        elif ',' in str(status):
            return 'purple'  # Multiple issues
        else:
            return 'gray'
    
    map_data['Color'] = map_data['Status'].apply(get_status_color)
    
    # Create the map
    fig = go.Figure()
    
    # Add points for each status
    unique_statuses = map_data['Status'].unique()
    
    for status in unique_statuses:
        status_data = map_data[map_data['Status'] == status]
        
        color = get_status_color(status)
        
        fig.add_trace(go.Scattermap(
            lat=status_data[lat_col],
            lon=status_data[lon_col],
            mode='markers',
            marker=dict(
                size=8,
                color=color
            ),
            text=status_data.apply(
                lambda row: f"Well ID: {row.get('API_UWI', 'N/A')}<br>"
                           f"Status: {row['Status']}<br>"
                           f"Total Months: {row['Total_Months']}<br>"
                           f"Coordinates: {row[lat_col]:.4f}, {row[lon_col]:.4f}",
                axis=1
            ),
            name=status,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Calculate center coordinates
    center_lat = map_data[lat_col].mean()
    center_lon = map_data[lon_col].mean()
    
    # Update layout
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8
        ),
        height=600,
        margin=dict(r=0, t=0, l=0, b=0)
    )
    
    return fig

# Sidebar for file uploads
st.sidebar.header("📁 Data Upload")

# Production data upload
production_file = st.sidebar.file_uploader(
    "Upload Production History Data (CSV)",
    type=['csv'],
    key="production_upload"
)

# Header data upload
header_file = st.sidebar.file_uploader(
    "Upload Well Header Data (CSV)",
    type=['csv'],
    key="header_upload"
)

# Analysis settings
st.sidebar.header("⚙️ Analysis Settings")

# Outlier detection settings
outlier_sensitivity = st.sidebar.slider(
    "Outlier Detection Sensitivity",
    min_value=1.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Higher values = fewer outliers detected. Lower values = more outliers detected."
)

st.sidebar.markdown("**Sensitivity Guide:**")
st.sidebar.markdown("• 1.0-1.5: Strict (more outliers)")
st.sidebar.markdown("• 1.5-2.0: Standard")
st.sidebar.markdown("• 2.0-3.0: Lenient (fewer outliers)")

# R² score filter for decline curve analysis
st.sidebar.header("📉 Decline Curve Filters")
r2_threshold = st.sidebar.number_input(
    "Minimum R² Score",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Only include wells with R² score above this threshold"
)

st.sidebar.markdown(f"**Current R² Filter: {r2_threshold:.2f}**")
st.sidebar.markdown("• 0.8+: High quality fits")
st.sidebar.markdown("• 0.6+: Moderate quality fits")
st.sidebar.markdown("• 0.4+: Lower quality fits")

# Process uploaded files
if production_file is not None:
    try:
        st.session_state.production_data = pd.read_csv(production_file)
        st.sidebar.success(f"✅ Production data loaded: {len(st.session_state.production_data)} rows")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading production data: {str(e)}")

if header_file is not None:
    try:
        st.session_state.header_data = pd.read_csv(header_file)
        st.sidebar.success(f"✅ Header data loaded: {len(st.session_state.header_data)} rows")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading header data: {str(e)}")

# Main content
if st.session_state.production_data is not None:
    
    # Process data button
    if st.button("🔄 Process Data", type="primary"):
        with st.spinner("Processing data..."):
            try:
                processed_data, prod_cols = process_production_data(st.session_state.production_data)
                st.session_state.processed_data = processed_data
                st.success("✅ Data processing completed!")
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
    
    # Show data analysis if processed
    if st.session_state.processed_data is not None:
        
        # Analyze well quality with custom outlier sensitivity
        well_analysis = analyze_well_quality(
            st.session_state.processed_data, 
            st.session_state.header_data,
            outlier_sensitivity=outlier_sensitivity
        )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🗺️ Interactive Map", "📈 Well Analysis", "📋 Data Quality", "📉 Decline Curve Analysis"])
        
        with tab1:
            st.header("📊 Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Production Data Summary")
                st.write(f"**Total Records:** {len(st.session_state.processed_data):,}")
                st.write(f"**Unique Wells:** {st.session_state.processed_data['API_UWI'].nunique() if 'API_UWI' in st.session_state.processed_data.columns else 'N/A'}")
                st.write(f"**Date Range:** {st.session_state.processed_data['ProducingMonth'].min().strftime('%Y-%m') if 'ProducingMonth' in st.session_state.processed_data.columns else 'N/A'} to {st.session_state.processed_data['ProducingMonth'].max().strftime('%Y-%m') if 'ProducingMonth' in st.session_state.processed_data.columns else 'N/A'}")
            
            with col2:
                if st.session_state.header_data is not None:
                    st.subheader("Header Data Summary")
                    st.write(f"**Total Wells:** {len(st.session_state.header_data):,}")
                    st.write(f"**Columns:** {len(st.session_state.header_data.columns)}")
                else:
                    st.info("No header data uploaded")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
        
        with tab2:
            st.header("🗺️ Interactive Well Map")
            
            if st.session_state.header_data is not None:
                # Create and display map
                map_fig = create_map_visualization(st.session_state.header_data, well_analysis)
                
                if map_fig is not None:
                    st.plotly_chart(map_fig, use_container_width=True)
                    
                    # Enhanced Legend with outlier column details
                    st.markdown("### Map Legend")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.markdown("🟢 **Normal Wells**")
                    with col2:
                        st.markdown("🔴 **Wells with Outliers**")
                    with col3:
                        st.markdown("🟠 **Wells with Null Values**")
                    with col4:
                        st.markdown("🔵 **Insufficient Data (<6 months)**")
                    with col5:
                        st.markdown("🟣 **Multiple Issues**")
                        
                    # Display outlier sensitivity impact
                    st.info(f"**Current Outlier Sensitivity: {outlier_sensitivity:.1f}** - Adjust in sidebar to reduce/increase outlier detection")
            else:
                st.info("📍 Upload header data with coordinate information to display the interactive map")
        
        with tab3:
            st.header("📈 Well Analysis")
            
            # Well quality summary
            if well_analysis:
                total_wells = len(well_analysis)
                outlier_wells = sum(1 for w in well_analysis.values() if w['has_outliers'])
                null_wells = sum(1 for w in well_analysis.values() if w['has_null_values'])
                insufficient_wells = sum(1 for w in well_analysis.values() if w['insufficient_data'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Wells", total_wells)
                with col2:
                    st.metric("Wells with Outliers", outlier_wells, f"{outlier_wells/total_wells*100:.1f}%")
                with col3:
                    st.metric("Wells with Null Values", null_wells, f"{null_wells/total_wells*100:.1f}%")
                with col4:
                    st.metric("Insufficient Data", insufficient_wells, f"{insufficient_wells/total_wells*100:.1f}%")
                
                # Detailed analysis table
                st.subheader("Detailed Well Analysis")
                
                # Convert analysis to DataFrame
                analysis_df = pd.DataFrame([
                    {
                        'Well_ID': w['well_id'],
                        'Status': w['status'],
                        'Total_Months': w['total_months'],
                        'Has_Outliers': w['has_outliers'],
                        'Has_Null_Values': w['has_null_values'],
                        'Insufficient_Data': w['insufficient_data'],
                        'Outlier_Columns': ', '.join(w['outlier_columns']) if w['outlier_columns'] else 'None',
                        'Null_Columns': ', '.join(w['null_columns']) if w['null_columns'] else 'None'
                    }
                    for w in well_analysis.values()
                ])
                
                # Filter options
                status_filter = st.selectbox(
                    "Filter by Status:",
                    ['All'] + list(analysis_df['Status'].unique())
                )
                
                if status_filter != 'All':
                    filtered_df = analysis_df[analysis_df['Status'] == status_filter]
                else:
                    filtered_df = analysis_df
                
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download processed data
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Well Analysis",
                    data=csv_buffer.getvalue(),
                    file_name="well_analysis.csv",
                    mime="text/csv"
                )
        
        with tab4:
            st.header("📋 Data Quality Report")
            
            # Overall data quality metrics
            st.subheader("Data Quality Metrics")
            
            # Missing values analysis
            missing_data = st.session_state.processed_data.isnull().sum()
            missing_percent = (missing_data / len(st.session_state.processed_data)) * 100
            
            quality_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Values': missing_data.values,
                'Missing_Percentage': missing_percent.values
            })
            
            # Filter and sort missing values data
            quality_df_filtered = quality_df[quality_df['Missing_Values'] > 0].copy()
            if len(quality_df_filtered) > 0:
                quality_df_filtered = quality_df_filtered.reset_index(drop=True)
                try:
                    quality_df_filtered = quality_df_filtered.sort_values('Missing_Values', ascending=False)
                except:
                    pass  # If sorting fails, show unsorted data
                st.subheader("Missing Values by Column")
                st.dataframe(quality_df_filtered, use_container_width=True)
            else:
                st.success("✅ No missing values detected in processed data")
            
            # Outlier detection summary with specific column analysis
            st.subheader("Outlier Detection Summary")
            
            # Calculate outlier statistics
            if well_analysis:
                total_wells = len(well_analysis)
                outlier_wells = sum(1 for w in well_analysis.values() if w['has_outliers'])
                null_wells = sum(1 for w in well_analysis.values() if w['has_null_values'])
                insufficient_wells = sum(1 for w in well_analysis.values() if w['insufficient_data'])
            else:
                total_wells = 0
                outlier_wells = 0
                null_wells = 0
                insufficient_wells = 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Sensitivity", f"{outlier_sensitivity:.1f}")
                st.metric("Wells with Outliers", outlier_wells)
                
            with col2:
                st.metric("Total Wells Analyzed", total_wells)
                outlier_percentage = (outlier_wells / total_wells * 100) if total_wells > 0 else 0
                st.metric("Outlier Wells %", f"{outlier_percentage:.1f}%")
            
            # Detailed outlier analysis by column
            st.subheader("Outlier Analysis by Column")
            
            # Create outlier summary by column
            outlier_by_column = {}
            for well_id, analysis in well_analysis.items():
                for outlier_detail in analysis.get('outlier_details', []):
                    column = outlier_detail.replace('Outlier-', '')
                    if column not in outlier_by_column:
                        outlier_by_column[column] = 0
                    outlier_by_column[column] += 1
            
            if outlier_by_column:
                outlier_df = pd.DataFrame([
                    {'Column': col, 'Wells_with_Outliers': count, 'Percentage': (count/total_wells*100)}
                    for col, count in outlier_by_column.items()
                ])
                outlier_df = outlier_df.sort_values('Wells_with_Outliers', ascending=False)
                st.dataframe(outlier_df, use_container_width=True)
                
                # Outlier reduction strategies
                st.subheader("Outlier Reduction Strategies")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Current Sensitivity: {:.1f}**".format(outlier_sensitivity))
                    st.markdown("**Methods to Reduce Outliers:**")
                    st.markdown("• Increase sensitivity to 2.0-2.5")
                    st.markdown("• Apply data smoothing techniques")
                    st.markdown("• Use percentile-based capping")
                    st.markdown("• Remove seasonal effects")
                    
                with col2:
                    st.markdown("**Column-Specific Recommendations:**")
                    for col, count in list(outlier_by_column.items())[:3]:  # Top 3 columns
                        percentage = count/total_wells*100
                        if percentage > 20:
                            st.markdown(f"• **{col}**: High outliers ({percentage:.1f}%) - Consider data validation")
                        elif percentage > 10:
                            st.markdown(f"• **{col}**: Moderate outliers ({percentage:.1f}%) - Review thresholds")
                        else:
                            st.markdown(f"• **{col}**: Low outliers ({percentage:.1f}%) - Normal variation")
            else:
                st.success("✅ No outliers detected with current sensitivity settings")
            
            production_cols = [
                'Prod_BOE', 'Prod_MCFE', 'GasProd_MCF', 'LiquidsProd_BBL',
                'WaterProd_BBL', 'RepGasProd_MCF'
            ]
            existing_cols = [col for col in production_cols if col in st.session_state.processed_data.columns]
            
            outlier_summary = []
            for col in existing_cols:
                try:
                    outliers, lb, ub = detect_outliers_iqr(st.session_state.processed_data, col)
                    outlier_summary.append({
                        'Column': col,
                        'Outliers_Count': len(outliers),
                        'Outliers_Percentage': f"{len(outliers)/len(st.session_state.processed_data)*100:.2f}%",
                        'Lower_Bound': f"{lb:.2f}",
                        'Upper_Bound': f"{ub:.2f}"
                    })
                except:
                    continue
            
            if outlier_summary:
                outlier_df = pd.DataFrame(outlier_summary)
                st.dataframe(outlier_df, use_container_width=True)
        
        with tab5:
            st.header("📉 Decline Curve Analysis & EUR Predictions")
            
            # Display current filter settings
            st.info(f"**Analysis Filters:** R² ≥ {r2_threshold:.2f} | Minimum 6 months of data")
            
            # Fit decline models button
            if st.button("🔬 Run Decline Curve Analysis", type="primary"):
                with st.spinner("Fitting decline curve models..."):
                    try:
                        # Fit decline models with R² threshold
                        decline_results = fit_decline_models(st.session_state.processed_data, r2_threshold=r2_threshold)
                        st.session_state.decline_results = decline_results
                        st.session_state.r2_threshold_used = r2_threshold
                        
                        if len(decline_results) > 0:
                            st.success(f"✅ Successfully analyzed {len(decline_results)} wells with R² ≥ {r2_threshold:.2f}")
                            st.info(f"📊 Found {len(decline_results)} wells meeting quality criteria")
                        else:
                            st.warning(f"⚠️ No wells met the criteria (R² ≥ {r2_threshold:.2f}, minimum 6 months of data)")
                            st.info("💡 Try lowering the R² threshold in the sidebar")
                    except Exception as e:
                        st.error(f"❌ Error in decline analysis: {str(e)}")
            
            # Show results if available
            if 'decline_results' in st.session_state and len(st.session_state.decline_results) > 0:
                decline_df = st.session_state.decline_results
                
                # Create sub-tabs for different analyses
                subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
                    "🏆 Top 20 Wells - 3 Month Forecast", 
                    "📊 EUR Analysis", 
                    "🗺️ Top 10 Wells Map", 
                    "📈 Individual Well Analysis",
                    "🔍 Model Validation & Accuracy"
                ])
                
                with subtab1:
                    st.subheader("🏆 Top 20 Highest Performing Wells - Next 3 Months Prediction")
                    
                    # Sort by EUR and get top 20
                    top_20_wells = decline_df.nlargest(20, 'EUR_BOE').copy()
                    
                    # Calculate 3-month predictions
                    predictions = []
                    for idx, row in top_20_wells.iterrows():
                        try:
                            t_future, q_future = predict_future_production(row, months_ahead=3)
                            total_prediction = np.sum(q_future)
                            predictions.append(total_prediction)
                        except:
                            predictions.append(0)
                    
                    top_20_wells['3_Month_Prediction_BOE'] = predictions
                    
                    # Display table with well names and predictions
                    display_cols = ['WellName', 'API_UWI', 'model', 'EUR_BOE', '3_Month_Prediction_BOE', 'R2', 'MAE', 'RMSE']
                    # Use available columns
                    available_cols = [col for col in display_cols if col in top_20_wells.columns]
                    if 'WellName' not in available_cols:
                        available_cols = ['API_UWI'] + [col for col in available_cols if col != 'API_UWI']
                    
                    top_20_display = top_20_wells[available_cols].copy()
                    
                    # Create proper column names
                    if 'WellName' in available_cols:
                        column_names = ['Well Name', 'Well ID', 'Best Model', 'EUR (BOE)', '3-Month Forecast (BOE)', 'R²', 'MAE', 'RMSE']
                    else:
                        column_names = ['Well ID', 'Best Model', 'EUR (BOE)', '3-Month Forecast (BOE)', 'R²', 'MAE', 'RMSE']
                    
                    top_20_display.columns = column_names[:len(available_cols)]
                    
                    # Format numbers
                    top_20_display['EUR (BOE)'] = top_20_display['EUR (BOE)'].round(0).astype(int)
                    top_20_display['3-Month Forecast (BOE)'] = top_20_display['3-Month Forecast (BOE)'].round(0).astype(int)
                    top_20_display['R²'] = top_20_display['R²'].round(3)
                    top_20_display['MAE'] = top_20_display['MAE'].round(2)
                    top_20_display['RMSE'] = top_20_display['RMSE'].round(2)
                    
                    st.dataframe(top_20_display, use_container_width=True)
                    
                    # Well selection for detailed view
                    st.subheader("📈 Click on a Well for Detailed Decline Curve")
                    
                    # Create well selection options with proper names
                    if 'WellName' in top_20_wells.columns:
                        well_options = []
                        for idx, row in top_20_wells.iterrows():
                            well_name = row['WellName']
                            well_id = row.get('API_UWI', row.get('well_id', ''))
                            well_options.append(f"{well_name} ({well_id})")
                        
                        selected_well_option = st.selectbox(
                            "Select a well to view its decline curve:",
                            options=well_options,
                            key="top20_well_selector"
                        )
                        
                        if selected_well_option:
                            # Extract well name from selection
                            selected_well_name = selected_well_option.split(' (')[0]
                            well_data = top_20_wells[top_20_wells['WellName'] == selected_well_name].iloc[0]
                    else:
                        selected_well = st.selectbox(
                            "Select a well to view its decline curve:",
                            options=top_20_wells['API_UWI'].tolist(),
                            key="top20_well_selector"
                        )
                        
                        if selected_well:
                            well_data = top_20_wells[top_20_wells['API_UWI'] == selected_well].iloc[0]
                    
                    # Check if a well was selected and display analysis
                    well_data = None
                    if 'WellName' in top_20_wells.columns and 'selected_well_option' in locals() and selected_well_option:
                        selected_well_name = selected_well_option.split(' (')[0]
                        well_data = top_20_wells[top_20_wells['WellName'] == selected_well_name].iloc[0]
                    elif 'selected_well' in locals() and selected_well:
                        well_data = top_20_wells[top_20_wells['API_UWI'] == selected_well].iloc[0]
                    
                    if well_data is not None:
                        # Create decline curve plot
                        fig = go.Figure()
                        
                        # Original data points
                        fig.add_trace(go.Scatter(
                            x=well_data['t_data'],
                            y=well_data['q_data'],
                            mode='markers',
                            name='Historical Data',
                            marker=dict(color='blue', size=8)
                        ))
                        
                        # Fitted curve
                        fig.add_trace(go.Scatter(
                            x=well_data['t_data'],
                            y=well_data['q_pred'],
                            mode='lines',
                            name=f'Fitted {well_data["model"].title()} Model',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Future prediction
                        try:
                            t_future, q_future = predict_future_production(well_data, months_ahead=12)
                            fig.add_trace(go.Scatter(
                                x=t_future,
                                y=q_future,
                                mode='lines',
                                name='12-Month Forecast',
                                line=dict(color='green', width=2, dash='dash')
                            ))
                        except:
                            pass
                        
                        # Create title with proper well identification
                        if 'WellName' in well_data and pd.notna(well_data['WellName']):
                            well_title = f"Decline Curve Analysis - {well_data['WellName']}"
                            if 'API_UWI' in well_data and pd.notna(well_data['API_UWI']):
                                well_title += f" ({well_data['API_UWI']})"
                        else:
                            well_title = f"Decline Curve Analysis - {well_data['API_UWI']}"
                        
                        fig.update_layout(
                            title=well_title,
                            xaxis_title="Time (Months)",
                            yaxis_title="Production Rate (BOE/month)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("EUR (BOE)", f"{well_data['EUR_BOE']:,.0f}")
                        with col2:
                            st.metric("R² Score", f"{well_data['R2']:.3f}")
                        with col3:
                            st.metric("MAE", f"{well_data['MAE']:.2f}")
                        with col4:
                            st.metric("RMSE", f"{well_data['RMSE']:.2f}")
                
                with subtab2:
                    st.subheader("📊 EUR Analysis & Distribution")
                    
                    # EUR distribution plot
                    fig_eur = px.histogram(
                        decline_df, 
                        x='EUR_BOE', 
                        nbins=30,
                        title="EUR Distribution Across All Wells"
                    )
                    fig_eur.update_layout(height=400)
                    st.plotly_chart(fig_eur, use_container_width=True)
                    
                    # EUR statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average EUR", f"{decline_df['EUR_BOE'].mean():,.0f} BOE")
                    with col2:
                        st.metric("Median EUR", f"{decline_df['EUR_BOE'].median():,.0f} BOE")
                    with col3:
                        st.metric("Max EUR", f"{decline_df['EUR_BOE'].max():,.0f} BOE")
                    with col4:
                        st.metric("Total Wells", len(decline_df))
                    
                    # Model performance comparison
                    st.subheader("Model Performance by Type")
                    model_performance = decline_df.groupby('model').agg({
                        'R2': 'mean',
                        'MAE': 'mean',
                        'RMSE': 'mean',
                        'EUR_BOE': 'mean'
                    }).round(3)
                    st.dataframe(model_performance, use_container_width=True)
                
                with subtab3:
                    st.subheader("🗺️ Top 10 Wells Geographic Distribution")
                    
                    if st.session_state.header_data is not None:
                        # Get top 10 wells by EUR
                        top_10_wells = decline_df.nlargest(10, 'EUR_BOE')
                        
                        # Merge with header data for coordinates
                        if 'API_UWI' in st.session_state.header_data.columns:
                            map_data = st.session_state.header_data.merge(
                                top_10_wells[['API_UWI', 'EUR_BOE', 'model', 'R2']], 
                                on='API_UWI', 
                                how='inner'
                            )
                            
                            # Check for coordinate columns
                            lat_cols = [col for col in map_data.columns if 'lat' in col.lower()]
                            lon_cols = [col for col in map_data.columns if 'lon' in col.lower()]
                            
                            if lat_cols and lon_cols:
                                lat_col = lat_cols[0]
                                lon_col = lon_cols[0]
                                
                                # Create map
                                fig_map = go.Figure()
                                
                                fig_map.add_trace(go.Scattermap(
                                    lat=map_data[lat_col],
                                    lon=map_data[lon_col],
                                    mode='markers',
                                    marker=dict(
                                        size=map_data['EUR_BOE'] / map_data['EUR_BOE'].max() * 20 + 10,
                                        color=map_data['EUR_BOE'],
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="EUR (BOE)")
                                    ),
                                    text=map_data.apply(
                                        lambda row: f"Well: {row['API_UWI']}<br>"
                                                   f"EUR: {row['EUR_BOE']:,.0f} BOE<br>"
                                                   f"Model: {row['model']}<br>"
                                                   f"R²: {row['R2']:.3f}",
                                        axis=1
                                    ),
                                    hovertemplate='%{text}<extra></extra>'
                                ))
                                
                                # Calculate center
                                center_lat = map_data[lat_col].mean()
                                center_lon = map_data[lon_col].mean()
                                
                                fig_map.update_layout(
                                    map=dict(
                                        style="open-street-map",
                                        center=dict(lat=center_lat, lon=center_lon),
                                        zoom=8
                                    ),
                                    height=500,
                                    margin=dict(r=0, t=0, l=0, b=0)
                                )
                                
                                st.plotly_chart(fig_map, use_container_width=True)
                                
                                # Well selector for EUR graph
                                st.subheader("📈 Click on a Well for EUR Graph")
                                
                                if 'WellName' in top_10_wells.columns:
                                    map_well_options = []
                                    for idx, row in top_10_wells.iterrows():
                                        well_name = row['WellName']
                                        well_id = row.get('API_UWI', row.get('well_id', ''))
                                        map_well_options.append(f"{well_name} ({well_id})")
                                    
                                    selected_map_well_option = st.selectbox(
                                        "Select a well to view its EUR analysis:",
                                        options=map_well_options,
                                        key="map_well_selector"
                                    )
                                    selected_map_well = selected_map_well_option.split(' (')[0] if selected_map_well_option else None
                                else:
                                    selected_map_well = st.selectbox(
                                        "Select a well to view its EUR analysis:",
                                        options=top_10_wells['API_UWI'].tolist(),
                                        key="map_well_selector"
                                    )
                                
                                if selected_map_well:
                                    well_matches = top_10_wells[top_10_wells['API_UWI'] == selected_map_well]
                                    if len(well_matches) > 0:
                                        well_eur_data = well_matches.iloc[0]
                                        
                                        # Create EUR graph with prediction
                                        fig_eur_pred = go.Figure()
                                        
                                        # Historical data
                                        fig_eur_pred.add_trace(go.Scatter(
                                            x=well_eur_data['t_data'],
                                            y=well_eur_data['q_data'],
                                            mode='markers',
                                            name='Historical Production',
                                            marker=dict(color='blue', size=8)
                                        ))
                                        
                                        # Fitted model
                                        fig_eur_pred.add_trace(go.Scatter(
                                            x=well_eur_data['t_data'],
                                            y=well_eur_data['q_pred'],
                                            mode='lines',
                                            name='Model Fit',
                                            line=dict(color='red', width=2)
                                        ))
                                        
                                        # Future prediction
                                        try:
                                            t_future, q_future = predict_future_production(well_eur_data, months_ahead=24)
                                            fig_eur_pred.add_trace(go.Scatter(
                                                x=t_future,
                                                y=q_future,
                                                mode='lines',
                                                name='24-Month Forecast',
                                                line=dict(color='green', width=2, dash='dash')
                                            ))
                                        except:
                                            pass
                                        
                                        fig_eur_pred.update_layout(
                                            title=f"EUR Analysis & Prediction - {selected_map_well}",
                                            xaxis_title="Time (Months)",
                                            yaxis_title="Production Rate (BOE/month)",
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig_eur_pred, use_container_width=True)
                                        
                                        # Performance metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("EUR (BOE)", f"{well_eur_data['EUR_BOE']:,.0f}")
                                        with col2:
                                            st.metric("R² Score", f"{well_eur_data['R2']:.3f}")
                                        with col3:
                                            st.metric("MAE", f"{well_eur_data['MAE']:.2f}")
                                        with col4:
                                            st.metric("RMSE", f"{well_eur_data['RMSE']:.2f}")
                                    else:
                                        st.error(f"Well {selected_map_well} not found in top 10 wells data")
                            else:
                                st.error("No latitude/longitude columns found in header data")
                        else:
                            st.error("No API_UWI column found in header data for mapping")
                    else:
                        st.info("Upload header data with coordinate information to display the map")
                
                with subtab4:
                    st.subheader("📈 Individual Well Analysis")
                    
                    # Well selector
                    selected_analysis_well = st.selectbox(
                        "Select a well for detailed analysis:",
                        options=decline_df['API_UWI'].tolist(),
                        key="analysis_well_selector"
                    )
                    
                    if selected_analysis_well:
                        well_analysis_matches = decline_df[decline_df['API_UWI'] == selected_analysis_well]
                        if len(well_analysis_matches) > 0:
                            well_analysis_data = well_analysis_matches.iloc[0]
                            
                            # Create comprehensive analysis plot
                            fig_analysis = go.Figure()
                            
                            # Historical data
                            fig_analysis.add_trace(go.Scatter(
                                x=well_analysis_data['t_data'],
                                y=well_analysis_data['q_data'],
                                mode='markers',
                                name='Historical Data',
                                marker=dict(color='blue', size=8)
                            ))
                            
                            # Model fit
                            fig_analysis.add_trace(go.Scatter(
                                x=well_analysis_data['t_data'],
                                y=well_analysis_data['q_pred'],
                                mode='lines',
                                name=f'{well_analysis_data["model"].title()} Model',
                                line=dict(color='red', width=2)
                            ))
                            
                            # Extended prediction
                            try:
                                t_future, q_future = predict_future_production(well_analysis_data, months_ahead=36)
                                fig_analysis.add_trace(go.Scatter(
                                    x=t_future,
                                    y=q_future,
                                    mode='lines',
                                    name='36-Month Forecast',
                                    line=dict(color='green', width=2, dash='dash')
                                ))
                            except:
                                pass
                            
                            fig_analysis.update_layout(
                                title=f"Comprehensive Analysis - {selected_analysis_well}",
                                xaxis_title="Time (Months)",
                                yaxis_title="Production Rate (BOE/month)",
                                height=500
                            )
                            
                            st.plotly_chart(fig_analysis, use_container_width=True)
                            
                            # Detailed metrics
                            st.subheader("Performance Metrics")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("EUR (BOE)", f"{well_analysis_data['EUR_BOE']:,.0f}")
                                st.metric("Model Type", well_analysis_data['model'].title())
                            
                            with col2:
                                st.metric("R² Score", f"{well_analysis_data['R2']:.4f}")
                                st.metric("MAE", f"{well_analysis_data['MAE']:.3f}")
                            
                            with col3:
                                st.metric("RMSE", f"{well_analysis_data['RMSE']:.3f}")
                                if not pd.isna(well_analysis_data['b']):
                                    st.metric("b parameter", f"{well_analysis_data['b']:.3f}")
                                else:
                                    st.metric("b parameter", "N/A")
                        else:
                            st.error(f"Well {selected_analysis_well} not found in decline curve data")
                
                with subtab5:
                    st.subheader("🔍 Model Validation & Accuracy Analysis")
                    
                    # Well selector for validation
                    if 'WellName' in decline_df.columns:
                        validation_well_options = []
                        for idx, row in decline_df.iterrows():
                            well_name = row['WellName']
                            well_id = row.get('API_UWI', row.get('well_id', ''))
                            validation_well_options.append(f"{well_name} ({well_id})")
                        
                        selected_validation_well_option = st.selectbox(
                            "Select a well for detailed model validation:",
                            options=validation_well_options,
                            key="validation_well_selector"
                        )
                        
                        if selected_validation_well_option:
                            selected_validation_well_name = selected_validation_well_option.split(' (')[0]
                            validation_well_matches = decline_df[decline_df['WellName'] == selected_validation_well_name]
                            if len(validation_well_matches) > 0:
                                validation_well_data = validation_well_matches.iloc[0]
                    else:
                        selected_validation_well = st.selectbox(
                            "Select a well for detailed model validation:",
                            options=decline_df['API_UWI'].tolist(),
                            key="validation_well_selector"
                        )
                        
                        if selected_validation_well:
                            validation_well_matches = decline_df[decline_df['API_UWI'] == selected_validation_well]
                            if len(validation_well_matches) > 0:
                                validation_well_data = validation_well_matches.iloc[0]
                    
                    if 'validation_well_data' in locals():
                        # Calculate advanced metrics
                        metrics = calculate_advanced_metrics(validation_well_data['q_data'], validation_well_data['q_pred'])
                        residual_stats = calculate_residuals(validation_well_data['q_data'], validation_well_data['q_pred'])
                        eur_stats = calculate_cumulative_eur_error(validation_well_data)
                        
                        # Display well info
                        well_title = validation_well_data.get('WellName', validation_well_data['API_UWI'])
                        st.subheader(f"Model Validation for: {well_title}")
                        
                        # Advanced metrics display
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error")
                        with col2:
                            st.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Square Error")
                        with col3:
                            st.metric("MAPE (%)", f"{metrics['MAPE']:.1f}", help="Mean Absolute Percentage Error")
                        with col4:
                            st.metric("R² Score", f"{metrics['R2']:.3f}", help="Coefficient of Determination")
                        with col5:
                            bias_color = "inverse" if abs(metrics['Bias']) < 10 else "normal"
                            st.metric("Bias", f"{metrics['Bias']:.2f}", help="Systematic Error (+ = overestimation)")
                        
                        # EUR accuracy
                        st.subheader("EUR Prediction Accuracy")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Actual EUR", f"{eur_stats['actual_eur']:,.0f} BOE")
                        with col2:
                            st.metric("Predicted EUR", f"{eur_stats['predicted_eur']:,.0f} BOE")
                        with col3:
                            error_color = "inverse" if abs(eur_stats['eur_error_pct']) < 10 else "normal"
                            st.metric("EUR Error (%)", f"{eur_stats['eur_error_pct']:+.1f}%", help="+ = overestimated, - = underestimated")
                        
                        # Create validation plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Actual vs Predicted scatter plot
                            fig_scatter = go.Figure()
                            
                            # Perfect prediction line
                            min_val = min(np.min(validation_well_data['q_data']), np.min(validation_well_data['q_pred']))
                            max_val = max(np.max(validation_well_data['q_data']), np.max(validation_well_data['q_pred']))
                            fig_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Actual vs predicted points
                            fig_scatter.add_trace(go.Scatter(
                                x=validation_well_data['q_data'],
                                y=validation_well_data['q_pred'],
                                mode='markers',
                                name='Data Points',
                                marker=dict(color='blue', size=8, opacity=0.7)
                            ))
                            
                            fig_scatter.update_layout(
                                title="Actual vs Predicted Production",
                                xaxis_title="Actual Production (BOE/month)",
                                yaxis_title="Predicted Production (BOE/month)",
                                height=400
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with col2:
                            # Residual plot
                            fig_residual = go.Figure()
                            
                            # Zero line
                            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                            
                            # Residuals vs predicted
                            fig_residual.add_trace(go.Scatter(
                                x=validation_well_data['q_pred'],
                                y=residual_stats['residuals'],
                                mode='markers',
                                name='Residuals',
                                marker=dict(color='green', size=8, opacity=0.7)
                            ))
                            
                            fig_residual.update_layout(
                                title="Residual Analysis",
                                xaxis_title="Predicted Production (BOE/month)",
                                yaxis_title="Residuals (Actual - Predicted)",
                                height=400
                            )
                            st.plotly_chart(fig_residual, use_container_width=True)
                        
                        # Time series analysis
                        st.subheader("Time Series Validation")
                        fig_timeseries = go.Figure()
                        
                        # Actual data
                        fig_timeseries.add_trace(go.Scatter(
                            x=validation_well_data['t_data'],
                            y=validation_well_data['q_data'],
                            mode='markers+lines',
                            name='Actual Production',
                            line=dict(color='blue')
                        ))
                        
                        # Predicted data
                        fig_timeseries.add_trace(go.Scatter(
                            x=validation_well_data['t_data'],
                            y=validation_well_data['q_pred'],
                            mode='lines',
                            name='Model Prediction',
                            line=dict(color='red', width=2)
                        ))
                        
                        # Residuals (secondary y-axis)
                        fig_timeseries.add_trace(go.Scatter(
                            x=validation_well_data['t_data'],
                            y=residual_stats['residuals'],
                            mode='markers',
                            name='Residuals',
                            marker=dict(color='green', size=6),
                            yaxis='y2'
                        ))
                        
                        fig_timeseries.update_layout(
                            title="Production Time Series with Residuals",
                            xaxis_title="Time (Months)",
                            yaxis_title="Production Rate (BOE/month)",
                            yaxis2=dict(
                                title="Residuals",
                                overlaying='y',
                                side='right',
                                showgrid=False
                            ),
                            height=500
                        )
                        st.plotly_chart(fig_timeseries, use_container_width=True)
                        
                        # Model quality assessment
                        st.subheader("Model Quality Assessment")
                        
                        quality_assessment = []
                        
                        # R² assessment
                        if metrics['R2'] >= 0.9:
                            r2_status = "Excellent"
                            r2_color = "🟢"
                        elif metrics['R2'] >= 0.8:
                            r2_status = "Good"
                            r2_color = "🟡"
                        elif metrics['R2'] >= 0.6:
                            r2_status = "Fair"
                            r2_color = "🟠"
                        else:
                            r2_status = "Poor"
                            r2_color = "🔴"
                        
                        quality_assessment.append(f"{r2_color} **R² Score ({metrics['R2']:.3f})**: {r2_status} model fit")
                        
                        # MAPE assessment
                        if metrics['MAPE'] <= 10:
                            mape_status = "Excellent"
                            mape_color = "🟢"
                        elif metrics['MAPE'] <= 20:
                            mape_status = "Good"
                            mape_color = "🟡"
                        elif metrics['MAPE'] <= 30:
                            mape_status = "Fair"
                            mape_color = "🟠"
                        else:
                            mape_status = "Poor"
                            mape_color = "🔴"
                        
                        quality_assessment.append(f"{mape_color} **MAPE ({metrics['MAPE']:.1f}%)**: {mape_status} prediction accuracy")
                        
                        # EUR error assessment
                        if abs(eur_stats['eur_error_pct']) <= 10:
                            eur_status = "Excellent"
                            eur_color = "🟢"
                        elif abs(eur_stats['eur_error_pct']) <= 20:
                            eur_status = "Good"
                            eur_color = "🟡"
                        elif abs(eur_stats['eur_error_pct']) <= 30:
                            eur_status = "Fair"
                            eur_color = "🟠"
                        else:
                            eur_status = "Poor"
                            eur_color = "🔴"
                        
                        quality_assessment.append(f"{eur_color} **EUR Error ({eur_stats['eur_error_pct']:+.1f}%)**: {eur_status} EUR prediction")
                        
                        # Display assessment
                        for assessment in quality_assessment:
                            st.markdown(assessment)
                        
                        # Improvement recommendations
                        st.subheader("Model Improvement Recommendations")
                        
                        recommendations = []
                        
                        if metrics['R2'] < 0.8:
                            recommendations.append("• Consider alternative decline models (exponential vs hyperbolic)")
                            recommendations.append("• Check for data quality issues or outliers")
                        
                        if metrics['MAPE'] > 20:
                            recommendations.append("• Model shows high percentage errors - consider data preprocessing")
                            recommendations.append("• Investigate production interruptions or operational changes")
                        
                        if abs(metrics['Bias']) > 10:
                            if metrics['Bias'] > 0:
                                recommendations.append("• Model systematically overestimates - consider adjusting decline parameters")
                            else:
                                recommendations.append("• Model systematically underestimates - review initial production rates")
                        
                        if abs(eur_stats['eur_error_pct']) > 20:
                            recommendations.append("• EUR prediction needs improvement - consider longer historical data")
                            recommendations.append("• Review economic limit assumptions")
                        
                        # Residual pattern analysis
                        residual_variance = np.var(residual_stats['residuals'])
                        if residual_variance > np.var(validation_well_data['q_data']) * 0.1:
                            recommendations.append("• High residual variance suggests model may be missing important patterns")
                        
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(rec)
                        else:
                            st.success("✅ Model shows good performance across all metrics!")
                        
                        # Data export option
                        st.subheader("Export Validation Data")
                        validation_export_data = pd.DataFrame({
                            'Time_Months': validation_well_data['t_data'],
                            'Actual_Production': validation_well_data['q_data'],
                            'Predicted_Production': validation_well_data['q_pred'],
                            'Residuals': residual_stats['residuals'],
                            'Standardized_Residuals': residual_stats['std_residuals']
                        })
                        
                        csv_data = validation_export_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Validation Data",
                            data=csv_data,
                            file_name=f"validation_{well_title.replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
            else:
                st.info("🔬 Click 'Run Decline Curve Analysis' to start analyzing well performance and EUR predictions")

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the Oil & Gas Well Data Analysis Platform! 🛢️
    
    This platform helps you analyze well production data with the following features:
    
    ### 📊 **Data Processing**
    - Upload production history and header data (CSV format)
    - Automatic data cleaning and preprocessing
    - Missing value handling with median replacement
    - Outlier detection using IQR method
    
    ### 🗺️ **Interactive Mapping**
    - Visualize well locations on an interactive map
    - Color-coded markers for data quality indicators
    - Detailed well information on hover
    
    ### 📈 **Analysis Features**
    - Identify wells with outliers in production data
    - Detect null values and insufficient data
    - Calculate cumulative production metrics
    - Time-based feature engineering
    
    ### 🚀 **Getting Started**
    1. Upload your **Production History Data** CSV file using the sidebar
    2. Upload your **Well Header Data** CSV file (with latitude/longitude coordinates)
    3. Click **"Process Data"** to clean and analyze your data
    4. Explore the results in the different tabs
    
    **Note:** Make sure your header data contains latitude and longitude columns for map visualization.
    """)
    
    st.info("👈 Please upload your data files using the sidebar to get started!")
