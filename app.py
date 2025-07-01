import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

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

def analyze_well_quality(production_df, header_df=None):
    """Analyze well data quality and create labels"""
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
        
        # Check for outliers
        for col in existing_prod_cols:
            if len(well_data) > 0 and well_data[col].notna().sum() > 0:
                try:
                    outliers, _, _ = detect_outliers_iqr(well_data, col)
                    if len(outliers) > 0:
                        analysis['has_outliers'] = True
                        analysis['outlier_columns'].append(col)
                except:
                    continue
        
        # Determine status
        statuses = []
        if analysis['has_outliers']:
            statuses.append('Outliers')
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
    
    # Create color mapping
    color_map = {
        'Normal': 'green',
        'Outliers': 'red',
        'Null Values': 'orange',
        'Insufficient Data': 'blue'
    }
    
    # Handle mixed statuses
    map_data['Color'] = map_data['Status'].apply(
        lambda x: 'purple' if ',' in str(x) else color_map.get(x, 'gray')
    )
    
    # Create the map
    fig = go.Figure()
    
    # Add points for each status
    unique_statuses = map_data['Status'].unique()
    
    for status in unique_statuses:
        status_data = map_data[map_data['Status'] == status]
        
        color = 'purple' if ',' in str(status) else color_map.get(status, 'gray')
        
        fig.add_trace(go.Scattermapbox(
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
        mapbox=dict(
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
        
        # Analyze well quality
        well_analysis = analyze_well_quality(
            st.session_state.processed_data, 
            st.session_state.header_data
        )
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🗺️ Interactive Map", "📈 Well Analysis", "📋 Data Quality"])
        
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
                    
                    # Legend
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
            quality_df = quality_df[quality_df['Missing_Values'] > 0].sort_values('Missing_Values', ascending=False)
            
            if len(quality_df) > 0:
                st.subheader("Missing Values by Column")
                st.dataframe(quality_df, use_container_width=True)
            else:
                st.success("✅ No missing values detected in processed data")
            
            # Outlier detection summary
            st.subheader("Outlier Detection Summary")
            
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
