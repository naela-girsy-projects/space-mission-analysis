"""
Analysis functions for space mission data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_success_rate(df, year_column='Year', status_column='Mission_Status'):
    """
    Calculate the success rate of missions per year.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        year_column (str): Name of the year column
        status_column (str): Name of the status column
        
    Returns:
        pandas.DataFrame: Dataframe with success rates by year
    """
    # Count total missions per year
    total_missions = df.groupby(year_column).size()
    
    # Count successful missions per year
    successful_missions = df[df[status_column] == 'Success'].groupby(year_column).size()
    
    # Calculate success rate
    success_rate = (successful_missions / total_missions * 100).round(2)
    
    # Create a dataframe with the results
    results = pd.DataFrame({
        'Year': total_missions.index,
        'Total_Missions': total_missions.values,
        'Successful_Missions': successful_missions.values,
        'Success_Rate_Percentage': success_rate.values
    })
    
    return results


def calculate_agency_performance(df, company_column, status_column='Mission_Status'):
    """
    Calculate performance metrics for each space agency.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        company_column (str): Name of the company/agency column
        status_column (str): Name of the status column
        
    Returns:
        pandas.DataFrame: Dataframe with agency performance metrics
    """
    # Get total missions per agency
    total_missions = df.groupby(company_column).size()
    
    # Get successful missions per agency
    successful_missions = df[df[status_column] == 'Success'].groupby(company_column).size()
    
    # Calculate success rate
    success_rate = (successful_missions / total_missions * 100).round(2)
    
    # Create a dataframe with the results
    results = pd.DataFrame({
        'Agency': total_missions.index,
        'Total_Missions': total_missions.values,
        'Successful_Missions': successful_missions.reindex(total_missions.index).fillna(0).astype(int).values,
        'Success_Rate_Percentage': success_rate.values
    })
    
    # Sort by total missions in descending order
    results = results.sort_values('Total_Missions', ascending=False)
    
    return results


def analyze_cost_efficiency(df, company_column, cost_column='Cost_USD', status_column='Mission_Status'):
    """
    Analyze cost efficiency for each space agency.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        company_column (str): Name of the company/agency column
        cost_column (str): Name of the cost column
        status_column (str): Name of the status column
        
    Returns:
        pandas.DataFrame: Dataframe with cost efficiency metrics
    """
    # Filter out rows with missing cost data
    df_with_cost = df.dropna(subset=[cost_column])
    
    # Calculate average cost per mission for each agency
    avg_cost = df_with_cost.groupby(company_column)[cost_column].mean()
    
    # Calculate average cost for successful missions
    successful_missions = df_with_cost[df_with_cost[status_column] == 'Success']
    avg_cost_success = successful_missions.groupby(company_column)[cost_column].mean()
    
    # Calculate number of missions with cost data
    mission_count = df_with_cost.groupby(company_column).size()
    
    # Create a dataframe with the results
    results = pd.DataFrame({
        'Agency': avg_cost.index,
        'Average_Cost_Per_Mission': avg_cost.values.round(2),
        'Average_Cost_Per_Successful_Mission': avg_cost_success.reindex(avg_cost.index).fillna(0).values.round(2),
        'Missions_With_Cost_Data': mission_count.values
    })
    
    # Sort by average cost in ascending order
    results = results.sort_values('Average_Cost_Per_Mission')
    
    return results


def analyze_orbit_success(df, orbit_column='Orbit_Category', status_column='Mission_Status'):
    """
    Analyze success rates for different orbit types.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        orbit_column (str): Name of the orbit category column
        status_column (str): Name of the status column
        
    Returns:
        pandas.DataFrame: Dataframe with orbit success rates
    """
    # Get total missions per orbit type
    total_missions = df.groupby(orbit_column).size()
    
    # Get successful missions per orbit type
    successful_missions = df[df[status_column] == 'Success'].groupby(orbit_column).size()
    
    # Calculate success rate
    success_rate = (successful_missions / total_missions * 100).round(2)
    
    # Create a dataframe with the results
    results = pd.DataFrame({
        'Orbit_Type': total_missions.index,
        'Total_Missions': total_missions.values,
        'Successful_Missions': successful_missions.reindex(total_missions.index).fillna(0).astype(int).values,
        'Success_Rate_Percentage': success_rate.values
    })
    
    # Sort by total missions in descending order
    results = results.sort_values('Total_Missions', ascending=False)
    
    return results


def analyze_launch_sites(df, country_column='Country', site_column='Launch_Site', status_column='Mission_Status'):
    """
    Analyze performance of different launch sites.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        country_column (str): Name of the country column
        site_column (str): Name of the launch site column
        status_column (str): Name of the status column
        
    Returns:
        pandas.DataFrame: Dataframe with launch site performance
    """
    # Combine country and site for a full location identifier
    df['Full_Location'] = df[site_column] + ', ' + df[country_column]
    
    # Get total missions per launch site
    total_missions = df.groupby('Full_Location').size()
    
    # Get successful missions per launch site
    successful_missions = df[df[status_column] == 'Success'].groupby('Full_Location').size()
    
    # Calculate success rate
    success_rate = (successful_missions / total_missions * 100).round(2)
    
    # Create a dataframe with the results
    results = pd.DataFrame({
        'Launch_Site': total_missions.index,
        'Total_Missions': total_missions.values,
        'Successful_Missions': successful_missions.reindex(total_missions.index).fillna(0).astype(int).values,
        'Success_Rate_Percentage': success_rate.values
    })
    
    # Sort by total missions in descending order
    results = results.sort_values('Total_Missions', ascending=False)
    
    return results


def analyze_yearly_trends(df, year_column='Year', company_column=None, orbit_column=None):
    """
    Analyze trends in mission frequency and characteristics over time.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        year_column (str): Name of the year column
        company_column (str, optional): Name of the company/agency column
        orbit_column (str, optional): Name of the orbit column
        
    Returns:
        dict: Dictionary with various trend analyses
    """
    results = {}
    
    # Calculate total missions per year
    yearly_missions = df.groupby(year_column).size().reset_index(name='Mission_Count')
    results['yearly_mission_count'] = yearly_missions
    
    # Calculate missions per agency per year (if company column provided)
    if company_column:
        # Get top agencies by mission count
        top_agencies = df[company_column].value_counts().head(5).index.tolist()
        
        # Filter for only top agencies
        top_agency_data = df[df[company_column].isin(top_agencies)]
        
        # Calculate mission count per year per agency
        agency_yearly = pd.crosstab(top_agency_data[year_column], top_agency_data[company_column])
        results['agency_yearly_missions'] = agency_yearly
    
    # Calculate missions per orbit type per year (if orbit column provided)
    if orbit_column:
        # Get top orbit types by mission count
        top_orbits = df[orbit_column].value_counts().head(5).index.tolist()
        
        # Filter for only top orbit types
        top_orbit_data = df[df[orbit_column].isin(top_orbits)]
        
        # Calculate mission count per year per orbit type
        orbit_yearly = pd.crosstab(top_orbit_data[year_column], top_orbit_data[orbit_column])
        results['orbit_yearly_missions'] = orbit_yearly
    
    return results


def analyze_launch_frequency(df, date_column, company_column=None):
    """
    Analyze patterns in launch frequency (monthly, seasonal, etc.).
    
    Args:
        df (pandas.DataFrame): Input dataframe
        date_column (str): Name of the date column (must be in datetime format)
        company_column (str, optional): Name of the company/agency column
        
    Returns:
        dict: Dictionary with various frequency analyses
    """
    results = {}
    
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Ensure datetime format
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Drop rows with invalid dates
    df_copy = df_copy.dropna(subset=[date_column])
    
    # Extract month and day of week from date
    df_copy['Month'] = df_copy[date_column].dt.month
    df_copy['Day_of_Week'] = df_copy[date_column].dt.dayofweek  # Monday=0, Sunday=6
    
    # Monthly launch frequency
    monthly_launches = df_copy.groupby('Month').size().reset_index(name='Launch_Count')
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
        7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    monthly_launches['Month_Name'] = monthly_launches['Month'].map(month_names)
    monthly_launches = monthly_launches.sort_values('Month')  # Ensure chronological order
    results['monthly_frequency'] = monthly_launches

    # Day of week launch frequency
    dow_launches = df_copy.groupby('Day_of_Week').size().reset_index(name='Launch_Count')
    day_names = {
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    }
    dow_launches['Day_Name'] = dow_launches['Day_of_Week'].map(day_names)
    dow_launches = dow_launches.sort_values('Day_of_Week')
    results['weekday_frequency'] = dow_launches

    # If company_column is provided, get monthly trends by agency
    if company_column:
        top_agencies = df_copy[company_column].value_counts().head(5).index
        top_df = df_copy[df_copy[company_column].isin(top_agencies)]
        monthly_agency = pd.crosstab(top_df['Month'], top_df[company_column])
        monthly_agency.index = monthly_agency.index.map(month_names)
        results['monthly_by_agency'] = monthly_agency

    return results


def identify_recent_trends(df, year_column='Year', status_column='Mission_Status', 
                          company_column=None, orbit_column=None, n_years=10):
    """
    Identify trends in the most recent years of space missions.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        year_column (str): Name of the year column
        status_column (str): Name of the status column
        company_column (str, optional): Name of the company/agency column
        orbit_column (str, optional): Name of the orbit column
        n_years (int): Number of recent years to analyze
        
    Returns:
        dict: Dictionary with various recent trend analyses
    """
    results = {}
    
    # Get the most recent n years
    recent_years = sorted(df[year_column].unique())[-n_years:]
    recent_data = df[df[year_column].isin(recent_years)]
    
    # Calculate success rate trend
    success_rate_trend = calculate_success_rate(recent_data, year_column, status_column)
    results['success_rate_trend'] = success_rate_trend
    
    # Analyze agency performance trends if company column provided
    if company_column:
        # Get top agencies in recent years
        top_recent_agencies = recent_data[company_column].value_counts().head(5).index.tolist()
        
        # Calculate yearly mission count for top agencies
        agency_trend = pd.crosstab(recent_data[recent_data[company_column].isin(top_recent_agencies)][year_column], 
                                 recent_data[recent_data[company_column].isin(top_recent_agencies)][company_column])
        results['agency_trend'] = agency_trend
        
        # Calculate success rates for top agencies in recent years
        agency_success = {}
        for agency in top_recent_agencies:
            agency_data = recent_data[recent_data[company_column] == agency]
            success_rate = (agency_data[status_column] == 'Success').mean() * 100
            agency_success[agency] = round(success_rate, 2)
            
        results['agency_success_rates'] = agency_success
    
    # Analyze orbit type trends if orbit column provided
    if orbit_column:
        # Get top orbit types in recent years
        top_recent_orbits = recent_data[orbit_column].value_counts().head(5).index.tolist()
        
        # Calculate yearly mission count for top orbit types
        orbit_trend = pd.crosstab(recent_data[recent_data[orbit_column].isin(top_recent_orbits)][year_column], 
                                recent_data[recent_data[orbit_column].isin(top_recent_orbits)][orbit_column])
        results['orbit_trend'] = orbit_trend
        
    return results


def calculate_correlation_matrix(df, numerical_columns):
    """
    Calculate correlation matrix between numerical features.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        numerical_columns (list): List of numerical column names
        
    Returns:
        pandas.DataFrame: Correlation matrix
    """
    # Select only numerical columns
    numerical_data = df[numerical_columns].copy()
    
    # Calculate correlation matrix
    correlation_matrix = numerical_data.corr()
    
    return correlation_matrix


def payload_analysis(df, payload_column, orbit_column=None, status_column='Mission_Status'):
    """
    Analyze relationship between payload mass and mission characteristics.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        payload_column (str): Name of the payload mass column
        orbit_column (str, optional): Name of the orbit column
        status_column (str): Name of the status column
        
    Returns:
        dict: Dictionary with various payload analyses
    """
    results = {}
    
    # Filter out rows with missing payload data
    df_with_payload = df.dropna(subset=[payload_column])
    
    # Calculate basic statistics
    payload_stats = df_with_payload[payload_column].describe()
    results['payload_stats'] = payload_stats
    
    # Calculate success rate by payload range
    # Create payload bins
    bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
    labels = ['0-100', '100-500', '500-1000', '1000-5000', '5000-10000', '10000+']
    
    df_with_payload['Payload_Range'] = pd.cut(df_with_payload[payload_column], bins=bins, labels=labels)
    
    # Calculate success rate for each payload range
    payload_success = df_with_payload.groupby('Payload_Range')[status_column].apply(
        lambda x: (x == 'Success').mean() * 100).round(2)
    
    results['payload_success_rate'] = payload_success
    
    # If orbit column provided, analyze payload by orbit type
    if orbit_column:
        payload_by_orbit = df_with_payload.groupby(orbit_column)[payload_column].agg(['mean', 'median', 'count'])
        results['payload_by_orbit'] = payload_by_orbit
    
    return results