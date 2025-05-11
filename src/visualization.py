"""
Visualization functions for space mission analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import folium
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_plotting_style():
    """Set the style for matplotlib plots."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20


def plot_success_rate_over_time(success_rate_df, title="Mission Success Rate Over Time"):
    """
    Plot mission success rate over time.
    
    Args:
        success_rate_df (pandas.DataFrame): DataFrame with success rate data
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    fig, ax1 = plt.subplots()
    
    # Plot success rate
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Success Rate (%)', color=color)
    ax1.plot(success_rate_df['Year'], success_rate_df['Success_Rate_Percentage'], 
             color=color, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create a second y-axis for mission count
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of Missions', color=color)
    ax2.bar(success_rate_df['Year'], success_rate_df['Total_Missions'], 
            color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Rotate x-axis labels for better readability if many years
    if len(success_rate_df) > 10:
        plt.xticks(rotation=45)
    
    # Set title and adjust layout
    plt.title(title)
    fig.tight_layout()
    
    return fig


def plot_agency_performance(agency_df, top_n=10, title="Space Agency Performance"):
    """
    Plot performance metrics for top space agencies.
    
    Args:
        agency_df (pandas.DataFrame): DataFrame with agency performance data
        top_n (int): Number of top agencies to display
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Select top N agencies by mission count
    top_agencies = agency_df.head(top_n)
    
    fig, ax = plt.subplots()
    
    # Create bar positions
    x = np.arange(len(top_agencies))
    width = 0.35
    
    # Plot total missions
    ax.bar(x - width/2, top_agencies['Total_Missions'], width, label='Total Missions', color='skyblue')
    
    # Plot successful missions
    ax.bar(x + width/2, top_agencies['Successful_Missions'], width, label='Successful Missions', color='green')
    
    # Add success rate as text labels
    for i, rate in enumerate(top_agencies['Success_Rate_Percentage']):
        ax.text(i, top_agencies['Total_Missions'].iloc[i] + 5, f"{rate}%", 
                ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Space Agency')
    ax.set_ylabel('Number of Missions')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(top_agencies['Agency'], rotation=45, ha='right')
    ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_launch_sites_map(launch_site_df, min_launches=10):
    """
    Create an interactive map of launch sites.
    
    Args:
        launch_site_df (pandas.DataFrame): DataFrame with launch site data
        min_launches (int): Minimum number of launches for a site to be included
        
    Returns:
        folium.Map: Interactive map object
    """
    # Filter sites with at least min_launches
    filtered_sites = launch_site_df[launch_site_df['Total_Missions'] >= min_launches].copy()
    
    # Create a world map centered at (0, 0)
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
    
    # Define a colormap for success rates
    colormap = LinearSegmentedColormap.from_list('success_rate', ['red', 'yellow', 'green'])
    
    # Dictionary of major launch sites with their coordinates
    # This is a simplified approach; in a real project, you might use geocoding
    launch_site_coordinates = {
        'Cape Canaveral': (28.3922, -80.6077),
        'Kennedy Space Center': (28.5728, -80.6490),
        'Vandenberg': (34.7420, -120.5724),
        'Baikonur': (45.9648, 63.3051),
        'Plesetsk': (62.9271, 40.5723),
        'Jiuquan': (40.9605, 100.2916),
        'Taiyuan': (37.5112, 112.5808),
        'Xichang': (28.2463, 102.0275),
        'Wenchang': (19.6145, 110.9510),
        'Sriharikota': (13.7199, 80.2304),
        'Tanegashima': (30.3984, 130.9698),
        'Guiana Space Centre': (5.2322, -52.7693),
        'Mahia Peninsula': (-39.2595, 177.8646),
        'Wallops': (37.9366, -75.4655),
        'Vostochny': (51.8845, 128.3336)
    }
    
    # Add markers for each site
    for _, site in filtered_sites.iterrows():
        site_name = site['Launch_Site'].split(',')[0].strip()
        
        # Try to find coordinates for the site
        coords = None
        for key, value in launch_site_coordinates.items():
            if key in site_name:
                coords = value
                break
        
        # Skip if coordinates not found
        if coords is None:
            continue
            
        # Calculate marker size based on number of launches
        radius = 5 + (site['Total_Missions'] / filtered_sites['Total_Missions'].max() * 15)
        
        # Calculate color based on success rate
        success_rate = site['Success_Rate_Percentage'] / 100
        color = colormap(success_rate)
        color_hex = '#{:02x}{:02x}{:02x}'.format(
            int(color[0] * 255), 
            int(color[1] * 255), 
            int(color[2] * 255)
        )
        
        # Create popup text
        popup_text = f"""
        <b>{site['Launch_Site']}</b><br>
        Total Missions: {site['Total_Missions']}<br>
        Successful: {site['Successful_Missions']}<br>
        Success Rate: {site['Success_Rate_Percentage']}%
        """
        
        # Add marker
        folium.CircleMarker(
            location=coords,
            radius=radius,
            popup=folium.Popup(popup_text, max_width=300),
            color=color_hex,
            fill=True,
            fill_color=color_hex,
            fill_opacity=0.7
        ).add_to(m)
    
    return m


def plot_cost_comparison(cost_df, top_n=8, title="Cost Efficiency by Space Agency"):
    """
    Plot cost comparison across agencies.
    
    Args:
        cost_df (pandas.DataFrame): DataFrame with cost data
        top_n (int): Number of top agencies to display
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Filter for agencies with sufficient cost data
    filtered_df = cost_df[cost_df['Missions_With_Cost_Data'] >= 5].head(top_n)
    
    fig, ax = plt.subplots()
    
    # Sort by average cost
    filtered_df = filtered_df.sort_values('Average_Cost_Per_Mission')
    
    # Create bar chart
    bars = ax.barh(filtered_df['Agency'], filtered_df['Average_Cost_Per_Mission'] / 1e6, 
                  color='skyblue', alpha=0.7)
    
    # Add mission count as text
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f"n={filtered_df['Missions_With_Cost_Data'].iloc[i]}", 
                va='center')
    
    # Set labels and title
    ax.set_xlabel('Average Cost Per Mission (Million USD)')
    ax.set_ylabel('Space Agency')
    ax.set_title(title)
    
    # Format x-axis to show millions
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}M'))
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_orbit_success_rates(orbit_df, title="Success Rates by Orbit Type"):
    """
    Plot success rates for different orbit types.
    
    Args:
        orbit_df (pandas.DataFrame): DataFrame with orbit success data
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Sort by total missions
    orbit_df = orbit_df.sort_values('Total_Missions', ascending=False)
    
    fig, ax = plt.subplots()
    
    # Create positions for bars
    x = np.arange(len(orbit_df))
    width = 0.6
    
    # Create bars with color based on success rate
    bars = ax.bar(x, orbit_df['Success_Rate_Percentage'], width, 
                 color=plt.cm.viridis(orbit_df['Success_Rate_Percentage'] / 100))
    
    # Add total mission count as text
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, 10, 
                f"n={orbit_df['Total_Missions'].iloc[i]}", 
                ha='center', va='bottom', color='white', fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Orbit Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(orbit_df['Orbit_Type'])
    ax.set_ylim(0, 100)
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_yearly_trends(yearly_data, metrics=None, title="Space Mission Trends by Year"):
    """
    Plot multiple metrics over time from yearly space mission data.
    
    Args:
        yearly_data (pandas.DataFrame): DataFrame with yearly data
        metrics (list): List of metrics to plot (defaults to success rate and mission count)
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    if metrics is None:
        metrics = ['Success_Rate_Percentage', 'Total_Missions']
    
    fig, ax = plt.subplots()
    
    years = yearly_data['Year']
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        if metric in yearly_data.columns:
            ax.plot(years, yearly_data[metric], marker='o', color=colors[i], 
                   label=metric.replace('_', ' '))
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Rotate x-axis labels if many years
    if len(years) > 10:
        plt.xticks(rotation=45)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_mission_complexity(complexity_df, title="Mission Complexity and Success"):
    """
    Plot relationship between mission complexity factors and success rates.
    
    Args:
        complexity_df (pandas.DataFrame): DataFrame with complexity metrics
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots()
    
    # Create a scatter plot with complexity score vs success rate
    scatter = ax.scatter(
        complexity_df['Complexity_Score'], 
        complexity_df['Success_Rate_Percentage'],
        s=complexity_df['Total_Missions'] * 2,  # Size based on mission count
        c=complexity_df['Average_Cost_Per_Mission'],  # Color based on cost
        cmap='viridis',
        alpha=0.7
    )
    
    # Add a colorbar for cost
    cbar = plt.colorbar(scatter)
    cbar.set_label('Average Cost Per Mission (USD)')
    
    # Add labels for significant points
    for i, row in complexity_df.iterrows():
        if row['Total_Missions'] > 20 or row['Complexity_Score'] > 8:
            ax.annotate(row['Mission_Type'], 
                       (row['Complexity_Score'], row['Success_Rate_Percentage']),
                       xytext=(5, 5), textcoords='offset points')
    
    # Add trend line
    if len(complexity_df) > 2:
        z = np.polyfit(complexity_df['Complexity_Score'], complexity_df['Success_Rate_Percentage'], 1)
        p = np.poly1d(z)
        ax.plot(complexity_df['Complexity_Score'], p(complexity_df['Complexity_Score']), 
               linestyle='--', color='red', alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('Mission Complexity Score')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(title)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def create_interactive_timeline(mission_df):
    """
    Create an interactive timeline of space missions.
    
    Args:
        mission_df (pandas.DataFrame): DataFrame with mission data
        
    Returns:
        plotly.graph_objects.Figure: Interactive timeline figure
    """
    # Sort by date
    mission_df = mission_df.sort_values('Launch_Date')
    
    # Create color mapping for mission status
    color_map = {'Success': 'green', 'Failure': 'red', 'Partial Success': 'orange'}
    
    # Create timeline
    fig = px.scatter(
        mission_df,
        x='Launch_Date',
        y='Agency',
        color='Mission_Status',
        size='Payload_Mass',
        hover_name='Mission_Name',
        hover_data=['Launch_Vehicle', 'Orbit_Type', 'Mission_Type'],
        color_discrete_map=color_map,
        title='Interactive Space Mission Timeline'
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Launch Date',
        yaxis_title='Space Agency',
        legend_title='Mission Status',
        height=800,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    
    return fig


def plot_failure_analysis(failure_df, title="Mission Failure Analysis"):
    """
    Create visualizations for analyzing mission failures.
    
    Args:
        failure_df (pandas.DataFrame): DataFrame with failure data
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Failure causes
    failure_counts = failure_df['Failure_Cause'].value_counts().head(8)
    ax1.barh(failure_counts.index, failure_counts.values, color='crimson')
    ax1.set_xlabel('Number of Failures')
    ax1.set_ylabel('Failure Cause')
    ax1.set_title('Common Causes of Mission Failures')
    
    # Plot 2: Failure by mission phase
    phase_counts = failure_df['Failure_Phase'].value_counts()
    ax2.pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%',
           colors=plt.cm.Set3.colors, startangle=90)
    ax2.set_title('Failures by Mission Phase')
    
    # Add overall title
    plt.suptitle(title, fontsize=20)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_payload_visualization(payload_df):
    """
    Create interactive visualization of payload data.
    
    Args:
        payload_df (pandas.DataFrame): DataFrame with payload data
        
    Returns:
        plotly.graph_objects.Figure: Interactive payload visualization
    """
    # Create a scatter plot with payload mass vs orbit altitude
    fig = px.scatter(
        payload_df,
        x='Payload_Mass',
        y='Orbit_Altitude',
        color='Agency',
        size='Payload_Count',
        hover_name='Mission_Name',
        hover_data=['Launch_Date', 'Mission_Status', 'Mission_Type'],
        log_x=True,  # Log scale for payload mass
        title='Payload Mass vs Orbit Altitude by Agency'
    )
    
    # Add orbit bands as shapes
    orbit_bands = [
        {'name': 'LEO', 'min': 160, 'max': 2000, 'color': 'rgba(200, 200, 255, 0.2)'},
        {'name': 'MEO', 'min': 2000, 'max': 35786, 'color': 'rgba(200, 255, 200, 0.2)'},
        {'name': 'GEO', 'min': 35786, 'max': 35796, 'color': 'rgba(255, 200, 200, 0.2)'}
    ]
    
    for band in orbit_bands:
        fig.add_shape(
            type='rect',
            x0=payload_df['Payload_Mass'].min() * 0.8,
            x1=payload_df['Payload_Mass'].max() * 1.2,
            y0=band['min'],
            y1=band['max'],
            fillcolor=band['color'],
            line=dict(width=0),
            layer='below',
            name=band['name']
        )
        
        # Add annotation for orbit band
        fig.add_annotation(
            x=payload_df['Payload_Mass'].min(),
            y=(band['min'] + band['max']) / 2,
            text=band['name'],
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=14)
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Payload Mass (kg, log scale)',
        yaxis_title='Orbit Altitude (km)',
        legend_title='Space Agency',
        height=700
    )
    
    return fig


def create_dashboard_components():
    """
    Create a set of visualization components for an interactive dashboard.
    
    Returns:
        dict: Dictionary of Plotly figures for dashboard
    """
    # This function would create a set of visualizations for a dashboard
    # In a real implementation, this would load and process data first
    
    # Placeholder for demonstration - in real use, these would be actual figures
    components = {
        'success_over_time': go.Figure(),
        'agency_comparison': go.Figure(),
        'launch_map': go.Figure(),
        'orbit_analysis': go.Figure(),
        'failure_breakdown': go.Figure(),
        'cost_analysis': go.Figure()
    }
    
    # Example of a simple component (would be populated with real data)
    components['success_over_time'] = go.Figure(
        data=[
            go.Scatter(x=[], y=[], name='Success Rate'),
            go.Bar(x=[], y=[], name='Mission Count', yaxis='y2')
        ],
        layout=go.Layout(
            title='Mission Success Rate Over Time',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Success Rate (%)', side='left'),
            yaxis2=dict(title='Mission Count', side='right', overlaying='y')
        )
    )
    
    return components


def plot_comparative_analysis(df1, df2, key_metric, title="Comparative Analysis"):
    """
    Create a comparative analysis visualization between two datasets.
    
    Args:
        df1 (pandas.DataFrame): First dataset
        df2 (pandas.DataFrame): Second dataset
        key_metric (str): The metric to compare
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    set_plotting_style()
    
    fig, ax = plt.subplots()
    
    # Get common categories between datasets
    common_categories = set(df1.index).intersection(set(df2.index))
    categories = sorted(list(common_categories))
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Extract values for the common categories
    values1 = [df1.loc[cat, key_metric] if cat in df1.index else 0 for cat in categories]
    values2 = [df2.loc[cat, key_metric] if cat in df2.index else 0 for cat in categories]
    
    # Create bars
    rects1 = ax.bar(x - width/2, values1, width, label=df1.name if hasattr(df1, 'name') else 'Dataset 1')
    rects2 = ax.bar(x + width/2, values2, width, label=df2.name if hasattr(df2, 'name') else 'Dataset 2')
    
    # Add labels and title
    ax.set_xlabel('Category')
    ax.set_ylabel(key_metric)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


def plot_mission_outcomes_sunburst(outcomes_df):
    """
    Create a sunburst chart showing mission outcomes by various factors.
    
    Args:
        outcomes_df (pandas.DataFrame): DataFrame with mission outcome data
        
    Returns:
        plotly.graph_objects.Figure: Interactive sunburst chart
    """
    # Ensure we have the required columns
    required_columns = ['Agency', 'Mission_Type', 'Launch_Vehicle', 'Mission_Status']
    if not all(col in outcomes_df.columns for col in required_columns):
        raise ValueError(f"outcomes_df must contain all these columns: {required_columns}")
    
    # Create sunburst chart
    fig = px.sunburst(
        outcomes_df,
        path=['Agency', 'Mission_Type', 'Launch_Vehicle', 'Mission_Status'],
        values='Count' if 'Count' in outcomes_df.columns else None,
        color='Mission_Status',
        color_discrete_map={'Success': 'green', 'Failure': 'red', 'Partial Success': 'orange'},
        branchvalues='total',
        title='Mission Outcomes by Agency, Type, and Vehicle'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(t=30, b=0, l=0, r=0),
        height=700,
        uniformtext=dict(minsize=10, mode='hide')
    )
    
    return fig