import pandas as pd 
import geopandas as gpd 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
from matplotlib.colors import LogNorm, Normalize
import os
from datetime import datetime
import requests
from xyzservices import TileProvider
from pyproj import Transformer
import json
from typing import Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from shapely.geometry import Point

class PlanetXYZ:
    def __init__(self, date_str, api_key=None):
        if api_key is None:
            api_key = os.environ.get('PLANET_API_KEY')
        if not api_key:
            raise ValueError("Planet API key not found. Set PLANET_API_KEY environment variable.")
        
        self.api_key = api_key
        self.date_str = date_str
        
        dt = pd.to_datetime(date_str)  # Check if the date string is valid
        # Fetch available mosaics
        year = dt.year
        month = f'{dt.month:02}'  # Ensure the month has a leading zero if less than 10
        # Find the most appropriate mosaic
        mosaic_id = f'global_monthly_{year}_{month}_mosaic'
        
        self.tile_provider = TileProvider({
            "url": f'https://tiles.planet.com/basemaps/v1/planet-tiles/{mosaic_id}/gmap/{{z}}/{{x}}/{{y}}.png?api_key={api_key}',
            "name": f"Planet {mosaic_id}",
            "attribution": "Planet Labs PBC",
            "cross_origin": "Anonymous"
        })

def load_data_dict(filename: str) -> Optional[Dict]:
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not a valid JSON file.")
        return None

def load_river_data(river_name, variable):
    df = pd.read_csv(f'data/{river_name}_recalculated_edited_width_standardized.csv')

    # Load the river_spatial_stats_FINAL dataframe
    stats_df = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
    
    # Get the 'tandem_date_start' and 'tandem_date_end' values for this river
    tandem_date_start = pd.to_datetime(stats_df.loc[stats_df['river_name'] == river_name, 'tandem_date_start'].iloc[0])
    tandem_date_end = pd.to_datetime(stats_df.loc[stats_df['river_name'] == river_name, 'tandem_date_end'].iloc[0])
    activity_year = pd.to_datetime(stats_df.loc[stats_df['river_name'] == river_name, 'activity_year'].iloc[0])
    
    river_data = df[df['river_name'] == river_name].dropna(subset=['dist_out', variable, 'node_id', 'reach_id'])
    river_data['dist_out'] = river_data['dist_out'].astype(float) / 1000
    river_data = river_data.sort_values('dist_out')
    
    # Add the tandem date information to the river_data
    river_data['tandem_date_start'] = tandem_date_start
    river_data['tandem_date_end'] = tandem_date_end
    river_data['activity_year'] = activity_year
    
    # Calculate the median meand_len
    median_meand_len = river_data['meand_len'].median()
    
    # Convert median_meand_len from meters to kilometers
    median_meand_len_km = median_meand_len / 1000
    
    # Calculate the window size based on the median meand_len
    window_size = int(median_meand_len_km / river_data['dist_out'].diff().median())
    
    # Ensure window_size is odd
    window_size = window_size if window_size % 2 != 0 else window_size + 1
    
    # Calculate the rolling mean
    river_data[f'{variable}_rolling'] = river_data[variable].rolling(window=window_size, center=True).mean()

    river_data['node_id'] = river_data['node_id'].astype(int)
    river_data['reach_id'] = river_data['reach_id'].astype(int)

    return river_data

def load_elevation_data(river_name):
    # List of parquet files to read
    parquet_files = [
        f'src/data/data/all_elevations_gdf_{river_name}.parquet'
        # f'src/data_handling/data/all_elevations_gdf_LILONGWE_extra.parquet'
    ]
    
    # Read and concatenate all parquet files
    dfs = []
    for file in parquet_files:
        try:
            df = gpd.read_parquet(file)
            dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: File {file} not found. Skipping.")
    
    if not dfs:
        raise ValueError("No parquet files were successfully read.")
    
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert to GeoDataFrame
    df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    
    # Convert to Web Mercator projection
    df = df.to_crs(epsg=3857)
    
    # Convert dist_out to kilometers
    df['dist_out'] = df['dist_out'].astype(float) / 1000
    
    # Function to get the center point of a group
    def get_center_point(group):
        center_index = len(group) // 2
        return group.iloc[center_index]
    
    # Group by node_id and reach_id, and select the center point for each group
    df_aggregated = df.groupby(['node_id', 'reach_id']).apply(get_center_point).reset_index(drop=True)
    
    df['node_id'] = df['node_id'].astype(int)
    df['reach_id'] = df['reach_id'].astype(int)

    return gpd.GeoDataFrame(df_aggregated, geometry='geometry', crs=df.crs)

def join_river_and_elevation_data(river_data, elevation_data):
    
    # Rename 'dist_out' columns before merging
    river_data = river_data.rename(columns={'dist_out': 'dist_out_river'})
    elevation_data = elevation_data.rename(columns={'dist_out': 'dist_out_elev'})
    
    # Merge the datasets based on node_id and reach_id
    merged_data = river_data.merge(elevation_data, on=['node_id', 'reach_id'], suffixes=('_river', '_elev'))
        
    # Use the geometry from the elevation data
    merged_data = gpd.GeoDataFrame(merged_data, geometry='geometry_elev', crs=elevation_data.crs)
    
    return merged_data


def plot_river_data(merged_data, variable, river_name, data_dict):
    # Set up the map projection
    proj = ccrs.epsg(3857)  # Web Mercator projection

    # Calculate the bounds of the data in EPSG:4326
    bounds = merged_data.to_crs(epsg=4326).total_bounds
    minx, miny, maxx, maxy = bounds

    # Convert bounds to Web Mercator
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    minx, miny = transformer.transform(minx, miny)
    maxx, maxy = transformer.transform(maxx, maxy)

    # Calculate width and height in meters
    width = maxx - minx
    height = maxy - miny

    # Prevent a zero division if height is zero
    if height == 0:
        print("Warning: computed height is zero; setting a minimal height to avoid division by zero.")
        height = 1  # or another appropriate minimal value

    # Now calculate the aspect ratio safely
    aspect_ratio = width / height

    # Set maximum dimensions (in inches) for 16-inch MacBook Pro
    max_width = 12  # Maximum width in inches
    max_height = 8  # Maximum height in inches

    # Calculate figure size based on aspect ratio and maximum dimensions
    if aspect_ratio > max_width / max_height:
        # Width limited
        fig_width = max_width
        fig_height = fig_width / aspect_ratio
    else:
        # Height limited
        fig_height = max_height
        fig_width = fig_height * aspect_ratio

    # Ensure minimum size
    fig_width = max(fig_width, 6)
    fig_height = max(fig_height, 4)

    # Create the figure and axis with the calculated size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw={'projection': proj})

    # Set the extent with some padding
    padding = 0.3  # 10% padding
    ax.set_extent([minx - width*padding, maxx + width*padding, 
                   miny - height*padding, maxy + height*padding], crs=proj)
    
    # Get the activity_year from the merged_data and convert to string
    activity_year = merged_data['activity_year'].iloc[0]
    activity_year_str = activity_year.strftime('%Y-%m')
    
    # Try to add Planet XYZ background, fall back to Google tiles if it fails
    try:
        planet_xyz = PlanetXYZ(date_str=activity_year_str)
        ctx.add_basemap(ax, source=planet_xyz.tile_provider, zoom=13, crs=proj)
        print("Successfully added Planet XYZ background.")
    except Exception as e:
        print(f"Failed to add Planet XYZ background: {e}")
        print("Falling back to Google Satellite tiles.")
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=13, crs=proj)

    # Filter out zero, negative, and non-finite values for the rolling average data
    valid_rolling_data = merged_data[(merged_data[f'{variable}_rolling'] > 0) & np.isfinite(merged_data[f'{variable}_rolling'])]
    
    if len(valid_rolling_data) == 0:
        print(f"Error: No valid positive data for {variable}'s rolling average.")
        plt.close(fig)
        return

    # Calculate 2nd and 98th percentiles for colormap range
    min_value = np.percentile(valid_rolling_data[f'{variable}_rolling'], 2)
    max_value = np.percentile(valid_rolling_data[f'{variable}_rolling'], 98)
    print(f'min_value: {min_value}, max_value: {max_value}')

    # Ensure min_value is not zero for LogNorm
    min_value = max(min_value, 0.01)

    # Use LogNorm for color scaling
    norm = LogNorm(vmin=min_value, vmax=max_value)
    cmap = plt.get_cmap('jet')

    # Plot the rolling mean
    scatter = ax.scatter(valid_rolling_data.geometry.x, valid_rolling_data.geometry.y,
                         c=valid_rolling_data[f'{variable}_rolling'], cmap=cmap,
                         s=20, alpha=1, transform=proj, norm=norm, 
                         edgecolor='none', linewidth=0.5)

    # Add a colorbar with manually specified ticks
    tick_locations = np.logspace(np.log10(min_value), np.log10(max_value), num=6)
    cbar = plt.colorbar(scatter, label=f'{variable} (rolling avg)', extend='both', ticks=tick_locations)
    cbar.ax.set_yticklabels([f'{x:.2f}' for x in tick_locations])

    # Calculate the correct position for lambda = 2
    lambda_2_pos = norm(2)

    # Add lambda = 2 annotation to the colorbar
    cbar.ax.axhline(y=2, color='black', linestyle='--', linewidth=1)
    cbar.ax.text(1.2, lambda_2_pos, 'λ = 2', va='center', ha='left', color='black', fontweight='bold', transform=cbar.ax.transAxes)

    # Set labels and title
    ax.set_title(f'{variable.capitalize()} along the {river_name} River')
    
    # Add north arrow
    ax.text(0.95, 0.05, 'N', transform=ax.transAxes, fontsize=20, 
            fontweight='bold', ha='center', va='center')
    ax.arrow(0.95, 0.05, 0, 0.05, transform=ax.transAxes, 
             fc='k', ec='k', head_width=0.03, head_length=0.03)
    
    # Add gridlines
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Plot avulsion and crevasse splay sites
    avulsion_sites = data_dict.get(river_name, {}).get("avulsion_lines", [])
    crevasse_splay_sites = data_dict.get(river_name, {}).get("crevasse_splay_lines", [])

    def find_nearest_point(df, target_dist):
        return df.iloc[(df['dist_out_river'] - target_dist).abs().argsort()[:1]]

    avulsion_points = []
    crevasse_points = []

    for site in avulsion_sites:
        nearest_point = find_nearest_point(merged_data, site['position'])
        avulsion_points.append({
            'geometry': Point(nearest_point.geometry.x.values[0], nearest_point.geometry.y.values[0]),
            'properties': {'date': site['date'], 'position': site['position']}
        })

    for site in crevasse_splay_sites:
        nearest_point = find_nearest_point(merged_data, site['position'])
        crevasse_points.append({
            'geometry': Point(nearest_point.geometry.x.values[0], nearest_point.geometry.y.values[0]),
            'properties': {'date': site['date'], 'position': site['position']}
        })

    if avulsion_points:
        avulsion_gdf = gpd.GeoDataFrame(avulsion_points, crs=merged_data.crs)
        avulsion_gdf['type'] = 'Avulsion'
        output_path = f'data/{river_name}_avulsion_sites.geojson'
        avulsion_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved avulsion sites to {output_path}")

    if crevasse_points:
        crevasse_gdf = gpd.GeoDataFrame(crevasse_points, crs=merged_data.crs)
        crevasse_gdf['type'] = 'Crevasse Splay'
        output_path = f'data/{river_name}_crevasse_splay_sites.geojson'
        crevasse_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved crevasse splay sites to {output_path}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

def save_rolling_average_geojson(merged_data, variable, river_name):
    # Filter out zero, negative, and non-finite values for the rolling average data
    valid_rolling_data = merged_data[(merged_data[f'{variable}_rolling'] > 0) & np.isfinite(merged_data[f'{variable}_rolling'])]
    
    # Select only the necessary columns
    columns_to_keep = ['node_id', 'reach_id', 'dist_out_river', f'{variable}_rolling', 'geometry_elev']
    geojson_data = valid_rolling_data[columns_to_keep]
    
    # Ensure the data is in EPSG:4326 (WGS84) for GeoJSON compatibility
    geojson_data = geojson_data.to_crs(epsg=4326)
    
    # Save to GeoJSON
    output_filename = f'data/{river_name}_{variable}_rolling_average.geojson'
    geojson_data.to_file(output_filename, driver='GeoJSON')
    print(f"Saved rolling average data to {output_filename}")

def save_sites_as_geojson(data_dict, river_name, merged_data):
    # Initialize empty lists for both types of sites
    avulsion_sites = []
    crevasse_splay_sites = []

    # Safely get the sites from the data_dict, defaulting to empty lists if not found
    if data_dict and river_name in data_dict:
        avulsion_sites = data_dict[river_name].get("avulsion_lines", [])
        crevasse_splay_sites = data_dict[river_name].get("crevasse_splay_lines", [])

    def find_nearest_point(df, target_dist):
        return df.iloc[(df['dist_out_river'] - target_dist).abs().argsort()[:1]]

    avulsion_points = []
    crevasse_points = []

    for site in avulsion_sites:
        nearest_point = find_nearest_point(merged_data, site['position'])
        avulsion_points.append({
            'geometry': Point(nearest_point.geometry.x.values[0], nearest_point.geometry.y.values[0]),
            'properties': {'date': site['date'], 'position': site['position']}
        })

    for site in crevasse_splay_sites:
        nearest_point = find_nearest_point(merged_data, site['position'])
        crevasse_points.append({
            'geometry': Point(nearest_point.geometry.x.values[0], nearest_point.geometry.y.values[0]),
            'properties': {'date': site['date'], 'position': site['position']}
        })

    if avulsion_points:
        avulsion_gdf = gpd.GeoDataFrame(avulsion_points, crs=merged_data.crs)
        avulsion_gdf['type'] = 'Avulsion'
        output_path = f'data/{river_name}_avulsion_sites.geojson'
        avulsion_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved avulsion sites to {output_path}")
    else:
        print(f"No avulsion sites found for {river_name}")

    if crevasse_points:
        crevasse_gdf = gpd.GeoDataFrame(crevasse_points, crs=merged_data.crs)
        crevasse_gdf['type'] = 'Crevasse Splay'
        output_path = f'data/{river_name}_crevasse_splay_sites.geojson'
        crevasse_gdf.to_file(output_path, driver='GeoJSON')
        print(f"Saved crevasse splay sites to {output_path}")
    else:
        print(f"No crevasse splay sites found for {river_name}")

    if not avulsion_points and not crevasse_points:
        print(f"No avulsion or crevasse splay sites found for {river_name}")

def main():
    river_name = '<RIVER_NAME>'
    variable = 'lambda'
    river_data = load_river_data(river_name, variable)
    elev_data = load_elevation_data(river_name)
    data_dict = load_data_dict('data/data_dict.json')
    merged_data = join_river_and_elevation_data(river_data, elev_data)
    
    # Save rolling average data as GeoJSON
    save_rolling_average_geojson(merged_data, variable, river_name)
    
    # Save avulsion and crevasse splay sites as GeoJSON
    save_sites_as_geojson(data_dict, river_name, merged_data)
    
    # Plot the data
    plot_river_data(merged_data, variable, river_name, data_dict)

if __name__ == '__main__':
    main()
