# Oil & Gas Well Data Analysis Platform

## Overview

This is a Streamlit-based web application designed for analyzing oil and gas well production data. The platform provides interactive data visualization, outlier detection, and data processing capabilities specifically tailored for petroleum industry datasets. The application allows users to upload CSV files containing well production data and perform comprehensive analysis with automated data cleaning and feature engineering.

## System Architecture

The application follows a simple single-file architecture pattern:

- **Frontend**: Streamlit web framework providing interactive UI components
- **Data Processing**: Pandas and NumPy for data manipulation and analysis
- **Visualization**: Plotly for interactive charts and graphs
- **Session Management**: Streamlit's built-in session state for data persistence across interactions

The architecture is designed as a monolithic application where all functionality is contained within a single Python file (`app.py`), making it easy to deploy and maintain for demonstration or prototype purposes.

## Key Components

### Data Upload and Management
- File upload interface supporting CSV formats
- Session state management for persistent data storage across user interactions
- Multiple data storage containers: `production_data`, `header_data`, and `processed_data`

### Data Processing Engine
- **Outlier Detection**: IQR (Interquartile Range) method for identifying statistical outliers
- **Data Cleaning**: Automatic removal of empty columns and handling of zero values
- **Feature Engineering**: Processing of production-specific metrics and calculations

### Visualization System
- Interactive charts using Plotly Express and Plotly Graph Objects
- Specialized visualizations for oil and gas production metrics
- Real-time data exploration capabilities

### Production Data Columns
The system is specifically designed to handle standard oil and gas production metrics including:
- BOE (Barrels of Oil Equivalent) production rates
- MCFE (Thousand Cubic Feet Equivalent) production
- Gas, liquids, and water production volumes
- Cumulative production data
- Daily production rates

## Data Flow

1. **Data Ingestion**: Users upload CSV files through the Streamlit interface
2. **Data Validation**: System checks for required columns and data integrity
3. **Data Cleaning**: 
   - Remove completely empty columns
   - Replace zero values with median values for production columns
   - Apply outlier detection using IQR method
4. **Data Processing**: Feature engineering and metric calculations
5. **Visualization**: Generate interactive charts and statistical summaries
6. **Analysis**: Provide insights and recommendations based on processed data

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive visualization library (both Express and Graph Objects)

### Python Standard Library
- **datetime**: Date and time handling for production data timestamps
- **io**: Input/output operations for file handling

## Deployment Strategy

The application is designed for simple deployment scenarios:

- **Development**: Local execution using `streamlit run app.py`
- **Cloud Deployment**: Compatible with Streamlit Cloud, Heroku, or similar platforms
- **Container Deployment**: Can be containerized using Docker for scalable deployment
- **Replit Deployment**: Optimized for Replit's cloud environment with automatic dependency management

The single-file architecture makes deployment straightforward with minimal configuration requirements.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Features

### Decline Curve Analysis & EUR Predictions (July 01, 2025)
- Added comprehensive decline curve modeling using Arps equations (exponential, harmonic, hyperbolic)
- Implemented EUR (Estimated Ultimate Recovery) calculations with performance metrics (MAE, RMSE, R²)
- Created Top 20 highest performing wells analysis with 3-month production forecasts
- Built interactive EUR analysis with distribution plots and model performance comparison
- Added Top 10 wells geographic map with EUR-based sizing and color coding
- Implemented individual well analysis with extended 36-month forecasts
- Integrated clickable well selection for detailed decline curve visualization
- Added outlier detection with configurable sensitivity controls (1.0-3.0 range)
- Enhanced outlier analysis showing specific columns with outliers (e.g., "Outlier-Prod_BOE")
- Implemented outlier reduction strategies and recommendations

### Model Validation & Accuracy Analysis (July 01, 2025)
- Added comprehensive model validation tab with actual vs predicted plots
- Implemented residual analysis with scatter plots to identify model patterns
- Created advanced error metrics including MAE, RMSE, MAPE, and Bias calculations
- Added EUR prediction accuracy analysis with percentage over/underestimation
- Built time series validation plots with dual y-axis for residuals
- Implemented automated model quality assessment with color-coded ratings
- Added specific improvement recommendations based on model performance
- Created data export functionality for validation results
- Enhanced well name display using "WellName" column when available
- Added R² score filtering system (adjustable threshold 0.0-1.0, default 0.8)

### EUR Error Distribution Analysis (July 01, 2025)
- Added comprehensive EUR error distribution analysis with binned error ranges (0-10%, 10-20%, 20-30%, 30-50%, 50-100%, >100%)
- Implemented EUR Prediction vs Actual scatter plot with R² score and mean error rate display
- Created color-coded scatter points based on error rate with interactive hover information
- Added perfect prediction reference line (y=x) for visual assessment
- Built error rate histogram showing frequency distribution of prediction errors
- Implemented distribution summary table showing count and percentage of wells in each error range
- Created interactive bar charts and pie charts for visual error distribution analysis
- Added model performance categorization (High/Medium/Low accuracy groups)
- Implemented overall model quality assessment with color-coded ratings
- Added key insights section with accuracy metrics and improvement recommendations
- Created export functionality for both summary and detailed error analysis data
- Built percentage breakdown display for each error range with total well counts

## Changelog

Changelog:
- July 01, 2025. Initial setup and decline curve analysis implementation