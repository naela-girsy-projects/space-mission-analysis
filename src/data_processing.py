"""
Data processing functions for space mission analysis project.
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime


def load_dataset(filepath):
    """
    Load the dataset from the specified filepath.
    
    Args:
        filepath (str): Path to the data file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    _, file_extension = os.path.splitext(filepath)
    
    if file_extension.lower() == '.csv':
        return pd.read_csv(filepath)
    elif file_extension.lower() == '.xlsx':
        return pd.read_excel(filepath)
    elif file_extension.lower() == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def standardize_date_format(df, date_column):
    """
    Standardize date format in the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        date_column (str): Name of the date column
        
    Returns:
        pandas.DataFrame: Dataframe with standardized date column
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert date strings to datetime objects
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    return df_copy


def extract_year(df, date_column, new_column_name='Year'):
    """
    Extract year from a date column and add it as a new column.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        date_column (str): Name of the date column
        new_column_name (str): Name for the new column containing years
        
    Returns:
        pandas.DataFrame: Dataframe with added year column
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Extract year from date column
    df_copy[new_column_name] = df_copy[date_column].dt.year
    
    return df_copy


def clean_company_names(df, company_column):
    """
    Standardize company names by removing suffixes and standardizing common variations.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        company_column (str): Name of the company column
        
    Returns:
        pandas.DataFrame: Dataframe with standardized company names
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Dictionary of company name replacements
    company_replacements = {
        r'SpaceX.*': 'SpaceX',
        r'NASA.*': 'NASA',
        r'RVSN USSR.*': 'RVSN USSR',
        r'General Dynamics.*': 'General Dynamics',
        r'CASC.*': 'CASC',
        r'Arianespace.*': 'Arianespace',
        r'ULA.*': 'ULA',
        r'Rocket Lab.*': 'Rocket Lab',
        r'.*Roscosmos.*': 'Roscosmos',
        r'.*Boeing.*': 'Boeing',
        r'.*Lockheed.*': 'Lockheed Martin',
        r'ISRO.*': 'ISRO',
        r'.*Blue Origin.*': 'Blue Origin',
        r'Virgin.*': 'Virgin Galactic',
    }
    
    # Apply replacements
    for pattern, replacement in company_replacements.items():
        df_copy[company_column] = df_copy[company_column].str.replace(pattern, replacement, regex=True)
    
    return df_copy


def categorize_mission_status(df, status_column, new_column_name='Mission_Status'):
    """
    Categorize mission status into Success, Failure, or Partial Success.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        status_column (str): Name of the status column
        new_column_name (str): Name for the new categorized status column
        
    Returns:
        pandas.DataFrame: Dataframe with categorized mission status
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Define patterns for different status categories
    success_patterns = ['successful', 'success', 'nominal', 'operational']
    failure_patterns = ['failure', 'failed', 'exploded', 'lost', 'crashed', 'unsuccessful']
    partial_patterns = ['partial', 'partial success', 'partial failure', 'anomaly']
    
    # Function to categorize status
    def categorize_status(status):
        if pd.isna(status):
            return 'Unknown'
        
        status_lower = str(status).lower()
        
        if any(pattern in status_lower for pattern in success_patterns):
            return 'Success'
        elif any(pattern in status_lower for pattern in failure_patterns):
            return 'Failure'
        elif any(pattern in status_lower for pattern in partial_patterns):
            return 'Partial Success'
        else:
            return 'Unknown'
    
    # Apply categorization
    df_copy[new_column_name] = df_copy[status_column].apply(categorize_status)
    
    return df_copy


def extract_cost_value(cost_string):
    """
    Extract numerical cost value from cost string.
    
    Args:
        cost_string (str): Cost string potentially containing currency symbols
        
    Returns:
        float: Extracted cost value
    """
    if pd.isna(cost_string):
        return np.nan
    
    # Convert to string if not already
    cost_string = str(cost_string)
    
    # Extract numbers using regex
    numbers = re.findall(r'[\d.]+', cost_string)
    
    if not numbers:
        return np.nan
    
    # Convert the first match to float
    try:
        value = float(numbers[0])
        
        # Adjust for million/billion
        if 'million' in cost_string.lower() or 'm' in cost_string.lower():
            value *= 1e6
        elif 'billion' in cost_string.lower() or 'b' in cost_string.lower():
            value *= 1e9
            
        return value
    except ValueError:
        return np.nan


def standardize_cost(df, cost_column, new_column_name='Cost_USD'):
    """
    Standardize cost values to a common currency (USD).
    
    Args:
        df (pandas.DataFrame): Input dataframe
        cost_column (str): Name of the cost column
        new_column_name (str): Name for the new standardized cost column
        
    Returns:
        pandas.DataFrame: Dataframe with standardized cost column
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Extract cost values
    df_copy[new_column_name] = df_copy[cost_column].apply(extract_cost_value)
    
    return df_copy


def categorize_orbit(df, orbit_column, new_column_name='Orbit_Category'):
    """
    Categorize orbits into standard categories.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        orbit_column (str): Name of the orbit column
        new_column_name (str): Name for the new categorized orbit column
        
    Returns:
        pandas.DataFrame: Dataframe with categorized orbit column
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Define orbit categories and their associated keywords
    orbit_categories = {
        'LEO': ['leo', 'low earth', 'low-earth'],
        'MEO': ['meo', 'medium earth', 'medium-earth'],
        'GEO': ['geo', 'geosynchronous', 'geostationary'],
        'HEO': ['heo', 'highly elliptical', 'high earth', 'high-earth', 'molniya'],
        'SSO': ['sso', 'sun synchronous', 'sun-synchronous', 'polar'],
        'Lunar': ['lunar', 'moon'],
        'Interplanetary': ['interplanetary', 'mars', 'venus', 'mercury', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto'],
        'Suborbital': ['suborbital', 'sub-orbital']
    }
    
    # Function to categorize orbits
    def categorize_orbit_type(orbit):
        if pd.isna(orbit):
            return 'Unknown'
            
        orbit_lower = str(orbit).lower()
        
        for category, keywords in orbit_categories.items():
            if any(keyword in orbit_lower for keyword in keywords):
                return category
                
        return 'Other'
    
    # Apply categorization
    df_copy[new_column_name] = df_copy[orbit_column].apply(categorize_orbit_type)
    
    return df_copy


def process_location_data(df, location_column, country_column_name='Country', site_column_name='Launch_Site'):
    """
    Extract country and launch site from location data.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        location_column (str): Name of the location column
        country_column_name (str): Name for the new country column
        site_column_name (str): Name for the new launch site column
        
    Returns:
        pandas.DataFrame: Dataframe with extracted location information
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Country mappings for standardization
    country_mappings = {
        'usa': 'USA',
        'united states': 'USA',
        'russia': 'Russia',
        'ussr': 'USSR',
        'china': 'China',
        'kazakhstan': 'Kazakhstan',
        'japan': 'Japan',
        'india': 'India',
        'france': 'France',
        'french guiana': 'French Guiana',
        'new zealand': 'New Zealand',
        'israel': 'Israel',
        'iran': 'Iran',
        'north korea': 'North Korea',
        'south korea': 'South Korea',
        'brazil': 'Brazil',
        'kenya': 'Kenya',
        'australia': 'Australia',
        'italy': 'Italy',
        'uk': 'United Kingdom',
        'united kingdom': 'United Kingdom',
        'germany': 'Germany'
    }
    
    # Function to extract country
    def extract_country(location):
        if pd.isna(location):
            return 'Unknown'
            
        location_lower = str(location).lower()
        
        for key, value in country_mappings.items():
            if key in location_lower:
                return value
                
        return 'Other'
    
    # Function to extract site name (everything except the country, if possible)
    def extract_site(location):
        if pd.isna(location):
            return 'Unknown'
        
        parts = str(location).split(',')
        return parts[0].strip() if parts else 'Unknown'
    
    # Apply extraction functions
    df_copy[country_column_name] = df_copy[location_column].apply(extract_country)
    df_copy[site_column_name] = df_copy[location_column].apply(extract_site)
    
    return df_copy


def handle_missing_values(df):
    """
    Handle missing values in the dataframe.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with handled missing values
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Fill missing categorical values
    categorical_columns = df_copy.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_copy[col] = df_copy[col].fillna('Unknown')
    
    # For numerical columns, leave NaN values for now (they'll be handled in specific analyses)
    
    return df_copy


def prepare_dataset(df, date_column, status_column, company_column=None, 
                   cost_column=None, orbit_column=None, location_column=None):
    """
    Complete data preparation pipeline.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        date_column (str): Name of the date column
        status_column (str): Name of the status column
        company_column (str, optional): Name of the company column
        cost_column (str, optional): Name of the cost column
        orbit_column (str, optional): Name of the orbit column
        location_column (str, optional): Name of the location column
        
    Returns:
        pandas.DataFrame: Fully processed dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Apply each processing step if applicable
    processed_df = standardize_date_format(processed_df, date_column)
    processed_df = extract_year(processed_df, date_column)
    processed_df = categorize_mission_status(processed_df, status_column)
    
    if company_column:
        processed_df = clean_company_names(processed_df, company_column)
    
    if cost_column:
        processed_df = standardize_cost(processed_df, cost_column)
    
    if orbit_column:
        processed_df = categorize_orbit(processed_df, orbit_column)
    
    if location_column:
        processed_df = process_location_data(processed_df, location_column)
    
    # Handle remaining missing values
    processed_df = handle_missing_values(processed_df)
    
    return processed_df