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

### Optimized Outlier Detection (July 01, 2025)
- Optimized outlier detection to focus only on essential modeling columns: Prod_BOE, Prod_MCFE, GasProd_MCF, LiquidsProd_BBL
- Removed non-essential columns (water production, administrative fields) from outlier analysis
- Enhanced data processing efficiency by targeting only columns critical for decline curve modeling
- Added clear documentation in Data Quality Report explaining which columns are analyzed and why
- Improved analysis speed and accuracy by eliminating noise from irrelevant data columns
- Maintained fallback support for datasets without primary production columns

### Enhanced Map Visualization & Statistical Analysis (July 01, 2025)
- Enhanced interactive map with well rankings, EUR-based sizing, and well names in hover tooltips
- Added ranked numbering (#1, #2, etc.) on map markers based on EUR performance
- Implemented EUR-sorted dropdown selections with error rate percentages displayed
- Added statistical confidence bands (1.65σ and 2.5σ) to decline curve analysis graphs
- Implemented outlier highlighting with red diamonds for points outside 2.5σ boundaries
- Enhanced EUR graphs with error rate percentages and outlier counts in titles
- Optimized performance by pre-sorting wells by EUR amounts for faster analysis
- Added comprehensive well selection dropdowns showing rank, EUR, model type, and R² scores
- Implemented statistical boundary lines matching industry standard confidence intervals
- Enhanced individual well analysis with same confidence interval visualization

### R² Score Distribution Analysis (July 01, 2025)
- Added comprehensive R² score distribution visualization with 0.05 increment ranges (0.00-0.05, 0.05-0.10, etc.)
- Implemented dual visualization: bar chart showing well counts and pie chart showing percentage distribution
- Created performance categorization: Excellent (R² ≥ 0.8), Good (0.7-0.8), Fair (0.5-0.7), Poor (<0.5)
- Added percentage breakdown showing what portion of wells fall into each performance category
- Built color-coded performance metrics with clear visual indicators for model quality assessment
- Implemented automatic model quality evaluation with success/warning messages based on R² thresholds
- Added data export functionality for R² distribution analysis results
- Created summary table showing well counts and percentages for each 0.05 R² range
- Enhanced insights section showing total wells in each performance tier with percentage calculations

### Security & Data Protection (July 01, 2025)
- Implemented comprehensive password protection system with username/password authentication
- Added secure login screen with hashed password verification for data protection
- Built session-based authentication with automatic session state management
- Created logout functionality that clears all sensitive data from memory
- Added security warnings and reminders in sidebar for confidential data handling
- Implemented authentication status indicators and session information display
- Enhanced data protection with clear security notices about sensitive oil & gas production data
- Added automatic data clearing on logout to prevent unauthorized access to uploaded files
- Built user-friendly security interface with proper error handling for failed login attempts

### Encrypted Excel File Support (July 01, 2025)
- Added support for password-protected Excel files (.xlsx, .xls) using msoffcrypto-tool
- Implemented dual file format support: regular CSV files and encrypted Excel files
- Built secure file decryption functionality with temporary file handling
- Added password input popup for encrypted Excel files with real-time decryption
- Implemented automatic cleanup of temporary files for security
- Created user-friendly interface for choosing between CSV and encrypted Excel formats
- Enhanced data security by supporting both file-level and application-level password protection
- Added progress indicators and error handling for decryption process
- Built seamless integration with existing data processing pipeline for both file types

### Enhanced Security & Memory Protection (July 01, 2025)
- Implemented hash-only credential storage (no plaintext passwords stored anywhere in Replit)
- Built secure in-memory file decryption with automatic temporary file cleanup
- Added secure password clearing functions to overwrite sensitive data in memory
- Enhanced logout functionality with complete session state clearing and garbage collection
- Implemented secure temporary directory handling with automatic cleanup
- Added memory buffer processing for encrypted files to minimize disk exposure
- Built comprehensive data clearing on logout to prevent any data persistence
- Enhanced security warnings and real-time password clearing after successful decryption
- Implemented multi-layer security: application auth + file encryption + memory protection

### Drilling Focus & Production Planning (July 01, 2025)
- Added comprehensive drilling focus analysis tab with production planning capabilities
- Implemented well performance metrics calculation including 30/90/180/365-day cumulative BOE
- Built EUR P10/P90 estimation using exponential decline curve fitting
- Added oil/gas percentage analysis and lateral length correlation studies
- Created top 10 wells performance table with customizable ranking metrics (EUR P10, 365-day cum, etc.)
- Implemented interactive performance map showing geographic distribution of top wells
- Built asset-level performance summary with reservoir/field grouping and averaging
- Added comprehensive forecasting system (1, 3, 6, 12 month production predictions)
- Created drilling focus insights with average lateral length, IP30, EUR P10, and oil content metrics
- Implemented performance distribution analysis with histogram visualizations
- Added export functionality for drilling focus data and asset performance summaries
- Built real-time well ranking system with selectable time periods for production planning

### Performance Optimization & Speed Improvements (July 01, 2025)
- Optimized encrypted Excel file decryption using direct in-memory processing (no temporary files)
- Enhanced CSV reading performance with C engine, low_memory=False, and vectorized operations
- Implemented fast date parsing with format hints and error handling fallbacks
- Optimized data processing with vectorized zero-value replacement and batch operations
- Added efficient cumulative calculations using grouped operations for better performance
- Built progress bars and performance timing feedback for user experience
- Enhanced file upload with performance indicators and processing time displays
- Optimized production column handling to focus only on essential modeling columns
- Implemented smart data type detection and optimized pandas operations
- Added fallback processing methods to ensure compatibility with various data formats

### Bug Fixes & Data Compatibility (July 01, 2025)
- Fixed critical 'WellId' column error in drilling focus functionality with flexible column mapping
- Added support for multiple well identifier column names: WellId, API_UWI, Well_ID, UWI, WellID
- Enhanced error handling with detailed debugging information for column availability
- Improved well performance analysis compatibility with different data formats
- Added comprehensive column validation and fallback mechanisms
- Built dynamic column detection for both production and header data sources
- Enhanced drilling focus calculations to work with various well naming conventions

## Changelog

Changelog:
- July 01, 2025. Initial setup and decline curve analysis implementation