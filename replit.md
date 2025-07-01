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

## Changelog

Changelog:
- July 01, 2025. Initial setup