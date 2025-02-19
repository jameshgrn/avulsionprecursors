import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List, Dict, Union
import warnings
import logging
import geopandas as gpd
import os
from datetime import datetime
from esda import Moran_Local
from libpysal.weights import DistanceBand
import scipy.stats as stats
import matplotlib.patches as patches  # imported to add black outlines to boxes

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Update Global Reach plot parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10  # Updated from 8
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 12  # Updated from 10
plt.rcParams['xtick.labelsize'] = 13  # Added
plt.rcParams['ytick.labelsize'] = 13  # Added
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# Update color palette
science_palette = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4']

def load_data(river_name: str) -> gpd.GeoDataFrame:
    logging.info(f"Loading data for {river_name}")

    df = pd.read_csv(f'data/{river_name}_recalculated_edited.csv')

    df = df[df['river_name'] == river_name].dropna(subset=['dist_out', 'lambda'])
    df['dist_out'] = df['dist_out'].astype(float) / 1000  # convert to km
    df = df.sort_values('dist_out', ascending=False)  # Sort descending
    
    # Clip negative values to 10^-2
    df['lambda'] = df['lambda'].clip(lower=1e-2)
    
    # Add lambda column
    epsilon = 1e-6
    df['lambda'] = df['lambda']
    df['lambda'] = df['lambda'].replace(-np.inf, np.log(epsilon))
    
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['dist_out'], np.zeros(len(df))),
        crs="EPSG:3857"
    )
    return gdf

def load_and_preprocess_data(river_names, lambda_c_values, data_dict):
    all_point_data = []
    all_random_data = []
    
    for river_name in river_names:
        try:
            gdf = load_data(river_name)
            river_data = lambda_c_values.get(river_name, {})
            lambda_c = river_data.get('lambda_c')
            tandem_date_start = river_data.get('tandem_date_start')
            
            if lambda_c is None:
                logging.warning(f"Missing lambda_c for {river_name}")
                continue
                
            # Reset index to avoid KeyError
            gdf = gdf.reset_index(drop=True)
            
            # Create masks for point and random data
            all_points = [point['position'] for point in data_dict.get(river_name, {}).get('avulsion_lines', []) + 
                         data_dict.get(river_name, {}).get('crevasse_splay_lines', [])
                         if point['date'] > tandem_date_start]
            
            if not all_points:
                logging.warning(f"No avulsion or crevasse points for {river_name}")
                continue
            
            # Create boolean mask instead of using indices
            event_mask = np.zeros(len(gdf), dtype=bool)
            for point in all_points:
                event_mask |= np.abs(gdf['dist_out'] - point) <= lambda_c
            
            # Split data using boolean mask
            point_data = gdf[event_mask].copy()
            random_data = gdf[~event_mask].copy()
            
            # Add river name column
            point_data['river_name'] = river_name
            random_data['river_name'] = river_name
            
            all_point_data.append(point_data)
            all_random_data.append(random_data)
            
        except Exception as e:
            logging.error(f"Error processing {river_name}: {str(e)}")
            continue
    
    if not all_point_data or not all_random_data:
        logging.warning("No data to process")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all data
    point_df = pd.concat(all_point_data, ignore_index=True)
    random_df = pd.concat(all_random_data, ignore_index=True)
    
    # Get the smaller sample size
    target_size = min(len(point_df), len(random_df))
    
    # Downsample both datasets to ensure equal sizes
    def downsample_proportionally(df, target_size):
        # Calculate proportions for each river
        river_props = df['river_name'].value_counts(normalize=True)
        # Calculate target samples for each river
        river_samples = (river_props * target_size).round().astype(int)
        
        # Adjust to exactly match target_size
        diff = target_size - river_samples.sum()
        if diff != 0:
            # Add or subtract the difference from the largest group(s)
            sorted_rivers = river_samples.sort_values(ascending=False)
            for i in range(abs(diff)):
                idx = i % len(sorted_rivers)
                river_samples[sorted_rivers.index[idx]] += 1 if diff > 0 else -1
        
        # Sample from each river
        sampled_df = pd.DataFrame()
        for river, n in river_samples.items():
            if n > 0:  # Only sample if we want at least one sample
                river_df = df[df['river_name'] == river]
                sampled = river_df.sample(n=min(n, len(river_df)), random_state=42)
                sampled_df = pd.concat([sampled_df, sampled])
        
        return sampled_df.reset_index(drop=True)
    
    # Downsample both datasets
    point_df = downsample_proportionally(point_df, target_size)
    random_df = downsample_proportionally(random_df, target_size)
    
    assert len(point_df) == len(random_df), "Sample sizes must be equal"
    
    return point_df, random_df

def load_data_dict(lambda_c_values: Dict[str, Dict[str, Union[float, str]]]) -> Dict:
    with open('data/data_dict.json', 'r') as file:
        data = json.load(file)
    
    for river, events in data.items():
        tandem_date_start = lambda_c_values.get(river, {}).get('tandem_date_start')
        if tandem_date_start:
            for event_type in ['avulsion_lines', 'crevasse_splay_lines']:
                data[river][event_type] = [
                    event for event in events.get(event_type, [])
                    if event['date'] and event['date'] > tandem_date_start
                ]
    
    return data

def load_lambda_c_values():
    """Load variogram range values from the CSV file."""
    filepath = 'data/manuscript_data/supplementary_table_1.csv'
    if not os.path.exists(filepath):
        logging.warning(f"Supplementary table file not found: {filepath}")
        return {}
    
    df = pd.read_csv(filepath)
    
    lambda_c_values = {}
    for _, row in df.iterrows():
        river = row['river_name']
        lambda_c_values[river] = {
            'lambda_c': row['variogram_range'] if pd.notnull(row['variogram_range']) else None,
            'tandem_date_start': row['tandem_date_start'] if pd.notnull(row['tandem_date_start']) else None,
            'L_A': row['L_A'] if pd.notnull(row['L_A']) else None
        }
    
    return lambda_c_values

def analyze_river(river_name: str, data_dict: Dict, lambda_c_values: Dict) -> Dict:
    df = load_data(river_name)
    
    river_data = lambda_c_values.get(river_name, {})
    lambda_c = river_data.get('lambda_c')
    tandem_date_start = river_data.get('tandem_date_start')
    
    if lambda_c is None:
        logging.warning(f"No lambda_c for {river_name}")
        return None
    
    avulsion_sites = data_dict.get(river_name, {}).get('avulsion_lines', [])
    crevasse_sites = data_dict.get(river_name, {}).get('crevasse_splay_lines', [])
    all_sites = [site for site in avulsion_sites + crevasse_sites if site['date'] > tandem_date_start]
    
    near_avulsion_lambda = []
    excluded_indices = set()
    
    for site in all_sites:
        site_position = site['position']
        local_df = df[np.abs(df['dist_out'] - site_position) <= lambda_c]
        if not local_df.empty:
            near_avulsion_lambda.extend(local_df['lambda'].tolist())
            excluded_indices.update(local_df.index)
    
    if not near_avulsion_lambda:
        return None

    global_lambda = df.loc[~df.index.isin(excluded_indices), 'lambda'].tolist()

    return {
        'river_name': river_name,
        'global_lambda': global_lambda,
        'near_avulsion_lambda': near_avulsion_lambda
    }

def calculate_normalized_weights(df):
    """Calculate normalized weights that sum to 1 for each group."""
    river_counts = df['river_name'].value_counts()
    weights = 1 / river_counts[df['river_name']].values
    # Normalize weights to sum to 1
    return weights / weights.sum()

def create_weighted_boxplot(point_data, random_data):
    # Calculate weights based on number of samples per river
    point_data['weight'] = calculate_normalized_weights(point_data)
    random_data['weight'] = calculate_normalized_weights(random_data)
    
    # Create figure with new dimensions
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Updated color scheme
    color1 = '#2b94c9'  
    color2 = '#c03672'  
    
    # Calculate weighted statistics for custom boxplot
    def get_box_stats(data, weights):
        median = weighted_quantile(data, 0.5, weights)
        q1 = weighted_quantile(data, 0.25, weights)
        q3 = weighted_quantile(data, 0.75, weights)
        iqr = q3 - q1
        whisker_low = max(q1 - 1.5 * iqr, data.min())
        whisker_high = min(q3 + 1.5 * iqr, data.max())
        mean = np.average(data, weights=weights)
        return median, q1, q3, whisker_low, whisker_high, mean
    
    # Get statistics for both groups
    global_stats = get_box_stats(random_data['lambda'], random_data['weight'])
    near_stats = get_box_stats(point_data['lambda'], point_data['weight'])
    
    # Modified custom box plotting for horizontal orientation with black outlines
    def plot_custom_box(pos, stats, color):
        median, q1, q3, wlow, whigh, mean = stats
        # Color filled box
        ax.fill_between([q1, q3], [pos-0.3, pos-0.3], [pos+0.3, pos+0.3], color=color, alpha=0.7)
        # Add black outline
        rect = patches.Rectangle((q1, pos-0.3), q3-q1, 0.6, fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        # Median line
        ax.vlines(median, pos-0.3, pos+0.3, color='black', linewidth=1)
        # Whiskers
        ax.hlines(pos, wlow, whigh, color='black', linewidth=0.5)
        ax.vlines([wlow, whigh], pos-0.15, pos+0.15, color='black', linewidth=0.5)
        # Mean marker - updated to use box color
        ax.plot(mean, pos, 'D', color=color, markeredgecolor='black', markersize=10)
    
    # Plot both boxes with new colors and black outlines
    plot_custom_box(0, global_stats, color1)
    plot_custom_box(1, near_stats, color2)
    
    # Modified swarmplot with increased sampling
    sampled_data = pd.concat([
        weighted_sample(random_data, n_samples=300).assign(Type='Global Reach'),
        weighted_sample(point_data, n_samples=300).assign(Type='Near Avulsion')
    ])
    
    sns.swarmplot(y='Type', x='lambda', data=sampled_data,
                  size=3, alpha=0.25, color='black', orient='h')
    ax.set_ylabel('')
    
    # Customize plot
    ax.set_xscale('log')
    ax.set_xlim(1e-3, ax.get_xlim()[1])  # Set minimum x limit to 10^-3
    ax.set_xlabel(r'$\Lambda$', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Removed the text annotations for stats so that stats are only printed to the terminal
    
    plt.tight_layout()
    plt.show()
    
    return random_data['lambda'], point_data['lambda']

# Add this helper function at the module level
def weighted_quantile(values, quantile, weights):
    """Calculate weighted quantile with proper handling of edge cases."""
    values = np.array(values)
    weights = np.array(weights)
    
    # Remove any nan values and corresponding weights
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]
    
    if len(values) == 0:
        return np.nan
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Sort values and weights together
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # Calculate cumulative weights
    cumsum = np.cumsum(sorted_weights)
    # Ensure we hit exactly 0 and 1
    cumsum = np.insert(cumsum, 0, 0)
    
    # Find index of quantile using interpolation
    idx = np.searchsorted(cumsum, quantile)
    if idx == 0:
        return sorted_values[0]
    if idx == len(sorted_values):
        return sorted_values[-1]
    
    # Interpolate if necessary
    if cumsum[idx] == quantile:
        return sorted_values[idx]
    else:
        # Linear interpolation between points
        return (sorted_values[idx-1] * (cumsum[idx] - quantile) +
                sorted_values[idx] * (quantile - cumsum[idx-1])) / \
               (cumsum[idx] - cumsum[idx-1])

def weighted_stats(data, weights):
    """Calculate weighted statistics."""
    data = np.array(data)
    weights = np.array(weights)
    
    # Remove any nan values
    mask = ~np.isnan(data)
    data = data[mask]
    weights = weights[mask]
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate statistics
    mean = np.average(data, weights=weights)
    median = weighted_quantile(data, 0.5, weights)
    q75 = weighted_quantile(data, 0.75, weights)
    q25 = weighted_quantile(data, 0.25, weights)
    iqr = q75 - q25
    
    return mean, median, iqr

def weighted_sample(df, n_samples=100):
    """Sample from dataframe using weights."""
    indices = np.random.choice(
        df.index, 
        size=min(n_samples, len(df)), 
        p=df['weight'],
        replace=False
    )
    return df.loc[indices]

def main():
    """Main function to run the analysis and create the figure."""
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Define river names to include
    river_names = ["B14_2", "VENEZ_2023", "VENEZ_2022_N", "ARG_LAKE_2", "ANJOBONY",
                   "RUVU_2", "V7_2", "V11_2", "TURKWEL", "MUSA", "BEMARIVO", "MANGOKY", "LILONGWE"]
    
    # Load data
    lambda_c_values = load_lambda_c_values()
    data_dict = load_data_dict(lambda_c_values)
    
    # Filter lambda_c_values and data_dict to include only the specified rivers
    lambda_c_values = {k: v for k, v in lambda_c_values.items() if k in river_names}
    data_dict = {k: v for k, v in data_dict.items() if k in river_names}
    
    # Debug: Print the filtered river names
    print("Processing the following rivers:", river_names)
    
    # Load and preprocess data once
    point_data, random_data = load_and_preprocess_data(river_names, lambda_c_values, data_dict)

    if point_data.empty or random_data.empty:
        logging.error("Insufficient data to create plot. Exiting.")
        return

    # Calculate weights once
    point_data['weight'] = calculate_normalized_weights(point_data)
    random_data['weight'] = calculate_normalized_weights(random_data)

    # Create plot using the weighted data
    sampled_random_lambda, sampled_point_lambda = create_weighted_boxplot(point_data, random_data)

    if sampled_random_lambda is None or sampled_point_lambda is None:
        logging.error("Failed to create plot. Exiting.")
        return

    # Print statistics to the terminal (to be later added in Illustrator)
    print("Statistics Summary:")
    print("===================")
    
    # Calculate weighted statistics
    global_mean, global_median, global_iqr = weighted_stats(random_data['lambda'], random_data['weight'])
    near_mean, near_median, near_iqr = weighted_stats(point_data['lambda'], point_data['weight'])
    
    print(f"Global Reach λ: mean={global_mean:.3f}, median={global_median:.3f}, IQR={global_iqr:.3f}")
    print(f"Near Avulsion λ: mean={near_mean:.3f}, median={near_median:.3f}, IQR={near_iqr:.3f}")
    print(f"Global lambda values: {len(random_data)}")
    print(f"Near-avulsion lambda values: {len(point_data)}")

    # Print lambda_c values
    print("\nLambda_c Values:")
    print("================")
    for river, data in lambda_c_values.items():
        lambda_c = data.get('lambda_c')
        if lambda_c is not None:
            print(f"{river}: lambda_c = {lambda_c:.2f}")
        else:
            print(f"{river}: lambda_c not available")

    # Print sample sizes
    print("\nSample sizes:")
    print("=============")
    print(f"Final sample sizes:")
    print(f"Global lambda values: {len(random_data)}")
    print(f"Near-avulsion lambda values: {len(point_data)}")

    # Save the figure data
    figure_data = {
        'boxplot_data': {
            'global': {
                'lambda': random_data['lambda'].tolist(),
                'river_name': random_data['river_name'].tolist(),
                'weight': random_data['weight'].tolist()
            },
            'near_avulsion': {
                'lambda': point_data['lambda'].tolist(),
                'river_name': point_data['river_name'].tolist(),
                'weight': point_data['weight'].tolist()
            }
        }
    }
    
    # Save to JSON with the expected filename
    with open('Fig4_plot1_data.json', 'w') as f:
        json.dump(figure_data, f)

if __name__ == "__main__":
    main()

