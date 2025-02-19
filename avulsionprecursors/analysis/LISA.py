import pandas as pd
import geopandas as gpd
import numpy as np
from libpysal.weights import DistanceBand
from esda import Moran_Local
import logging
from typing import Tuple, List, Dict
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from adjustText import adjust_text
from scipy.signal import savgol_filter

# Added imports for Earth Engine authentication
import ee
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_gee():
    """
    Initialize Google Earth Engine using service account credentials.
    Uses the updated JSON file path for authentication.
    """
    try:
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        credentials_path = "<YOUR_CREDENTIALS_PATH>"
        print(f"Initializing Google Earth Engine using credentials file at: {credentials_path}")
        if not os.path.isfile(credentials_path):
            print(f"Credentials file does not exist at: {credentials_path}")
            ee.Initialize()  # Fallback to default authentication if desired.
        else:
            credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
            ee.Initialize(credentials)
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")

def load_data(river_name: str) -> gpd.GeoDataFrame:
    logging.info(f"Loading data for {river_name}")

    df = pd.read_csv(f'data/{river_name}_recalculated_edited.csv')

        
    df = df[df['river_name'] == river_name].dropna(subset=['dist_out', 'lambda'])
    df['dist_out'] = df['dist_out'].astype(float) / 1000  # convert to km
    df = df.sort_values('dist_out', ascending=False)  # Sort descending
    
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

def smooth_data(data: np.ndarray, window_size: int, dist_array: np.ndarray) -> np.ndarray:
    """
    Smooth data using Savitzky-Golay filter where window_size corresponds to lambda_c
    
    Parameters:
    -----------
    data : np.ndarray
        Lambda values to smooth
    window_size : int
        Lambda_c in kilometers
    dist_array : np.ndarray
        Distance values in kilometers
        
    Returns:
    --------
    np.ndarray
        Smoothed lambda values
    """
    # Convert lambda_c from km to m
    window_size_m = window_size * 1000
    
    # Calculate point spacing in meters
    point_spacing_m = np.median(np.diff(dist_array * 1000))
    
    # Calculate number of points that fit in our lambda_c window
    points_in_window = int(window_size_m / point_spacing_m)
    
    # Ensure window is odd (required for Savitzky-Golay)
    if points_in_window % 2 == 0:
        points_in_window += 1
    
    # Ensure minimum window size of 3
    points_in_window = max(3, points_in_window)
    
    # Use a lower polynomial order for smoother results
    # Rule of thumb: polyorder should be significantly less than window length
    poly_order = min(2, points_in_window // 4)
    
    # Apply filter multiple times for smoother results
    smoothed = data.copy()
    for _ in range(3):  # Apply filter 3 times
        smoothed = savgol_filter(
            smoothed, 
            window_length=points_in_window, 
            polyorder=poly_order
        )
    
    logging.debug(f"Window points: {points_in_window}, Poly order: {poly_order}")
    
    return smoothed

def analyze_river(river_name: str, distances: Dict[str, float]) -> Dict[str, Tuple[gpd.GeoDataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Analyze a single river's spatial patterns using multiple characteristic distances
    Returns only smoothed LISA results.
    """
    # Load data
    gdf = load_data(river_name)
    
    results = {}
    for distance_type, lambda_c in distances.items():
        logging.info(f"River: {river_name}, Distance type: {distance_type}, Metrics: {lambda_c}")
        
        # Use lambda_c for smoothing
        smoothed_lambda = smooth_data(gdf['lambda'].values, window_size=int(lambda_c), dist_array=gdf['dist_out'].values)
        gdf[f'smoothed_lambda_{distance_type}'] = smoothed_lambda
        
        # Use dominant wavelength for LISA threshold
        threshold = distances['dominant_wavelength']  # Use dominant wavelength directly
        lisa_values, p_values, quadrants = calculate_lisa(gdf, f'smoothed_lambda_{distance_type}', threshold)
        
        results[distance_type] = (gdf, lisa_values, p_values, quadrants)
    
    return results

def load_characteristic_distances() -> Dict[str, Dict[str, float]]:
    """
    Load characteristic distances from supplementary table
    
    Returns:
    --------
    Dict[str, Dict[str, float]]
        Nested dictionary with river names as keys and distance metrics as values:
        {
            'river_name': {
                'variogram_range': float,
                'dominant_wavelength': float
            }
        }
    """
    supp_table = pd.read_csv('data/manuscript_data/supplementary_table_1.csv')
    
    distance_dict = {}
    for _, row in supp_table.iterrows():
        distance_dict[row['river_name']] = {
            'variogram_range': row['variogram_range'],
            'dominant_wavelength': row['dominant_wavelength'],
        }
    
    return distance_dict

def calculate_lisa(data: gpd.GeoDataFrame, variable: str, distance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Local Indicators of Spatial Association (LISA) using smoothed data directly
    """
    logging.info(f"Calculating LISA statistics with distance threshold: {distance}")
    
    try:
        # Create distance-based weights matrix
        w = DistanceBand.from_dataframe(
            data, 
            threshold=distance*2, 
            binary=False,
            alpha=-1,
            silence_warnings=True
        )
        
        # Handle isolated points
        if w.islands:
            logging.warning(f"Found {len(w.islands)} isolated points. Adding self-connections.")
            for island in w.islands:
                w[island] = {island: 1}
        
        # Use smoothed values directly
        values = data[variable].values
        
        # Calculate local Moran's I using smoothed values
        lisa = Moran_Local(
            values,
            w,
            transformation='r',
            permutations=999,
            geoda_quads=True,
            seed=42
        )
        
        # Log some statistics for verification
        logging.info(f"Mean LISA value: {np.mean(lisa.Is):.3f}")
        logging.info(f"Number of significant clusters: {np.sum(lisa.p_sim < 0.05)}")
        
        return lisa.Is, lisa.p_sim, lisa.q
        
    except Exception as e:
        logging.error(f"Error calculating LISA statistics: {str(e)}")
        raise

def plot_river_analysis(gdf: gpd.GeoDataFrame, lisa_values: np.ndarray, 
                       p_values: np.ndarray, quadrants: np.ndarray,
                       river_name: str, distance_type: str,
                       save_path: str = "figures"):
    """
    Create plot showing only smoothed LISA clusters without annotations
    """
    # Update base path to the full absolute path
    base_path = "data/manuscript_data/plots"
    os.makedirs(base_path, exist_ok=True)
    
    # Load event locations
    events = load_event_locations()
    river_events = events.get(river_name, {})
    
    fig, ax3 = plt.subplots(1, 1, figsize=(7, 3))
    
    # Plot LISA Clusters
    # GeoDa quadrant scheme:
    # 1 = HH (hot spot)
    # 2 = LH (spatial outlier)
    # 3 = LL (cold spot)
    # 4 = HL (spatial outlier)
    # 0 = not significant
    colors = ['gray', 'red', 'lightblue', 'blue', 'pink']
    labels = ['Not Significant', 'High-High', 'Low-Low', 'Low-High', 'High-Low']
    
    # Plot points directly using quadrants from Moran_Local
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = quadrants == i
        if np.any(mask):
            ax3.scatter(gdf['dist_out'][mask], gdf[f'smoothed_lambda_{distance_type}'][mask], 
                       c=color, label=label, alpha=0.6)
    
    # Add events to plot
    ymin, ymax = ax3.get_ylim()
    height = ymax - ymin
    for avulsion in river_events.get('avulsion_lines', []):
        pos = avulsion['position'] 
        ax3.axvline(x=pos, color='red', linestyle='--', alpha=1)
    
    for splay in river_events.get('crevasse_splay_lines', []):
        pos = splay['position'] 
        ax3.axvline(x=pos, color='green', linestyle=':', alpha=1)
    
    ax3.set_xlabel('Distance from Outlet(km)', fontsize=14)
    ax3.set_ylabel('Smoothed $\Lambda$', fontsize=14)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(loc='best', fontsize=8)
    ax3.set_yscale('log')
    ax3.invert_xaxis()

    plt.tight_layout()
    plt.show()

    plt.close('all')

def classify_lisa_clusters(lisa_values: np.ndarray, p_values: np.ndarray, 
                         alpha: float = 0.05) -> np.ndarray:
    """
    Classify LISA clusters according to Anselin's Local Moran's I
    
    Returns:
    --------
    np.ndarray
        Array of integers:
        0: Not Significant
        1: High-High (HH)
        2: Low-Low (LL)
        3: Low-High (LH)
        4: High-Low (HL)
    """
    # Initialize cluster array
    clusters = np.zeros(len(lisa_values), dtype=int)
    
    # Get significant locations
    sig_idx = p_values < alpha
    
    # Standardize the original values
    values_std = (lisa_values - np.mean(lisa_values)) / np.std(lisa_values)
    
    # For significant locations, classify based on original value and LISA statistic
    for i in np.where(sig_idx)[0]:
        if values_std[i] > 0:  # High value
            if lisa_values[i] > 0:
                clusters[i] = 1  # High-High
            else:
                clusters[i] = 4  # High-Low
        else:  # Low value
            if lisa_values[i] > 0:
                clusters[i] = 2  # Low-High
            else:
                clusters[i] = 3  # Low-Low
    
    return clusters

def load_event_locations() -> Dict[str, Dict]:
    """Load avulsion and crevasse splay locations from JSON"""
    with open('data/data_dict.json', 'r') as f:
        return json.load(f)

def plot_smoothing_comparison(gdf: gpd.GeoDataFrame, distance_type: str, river_name: str):
    """Plot raw vs smoothed data to verify smoothing behavior"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot raw data
    ax.scatter(gdf['dist_out'], gdf['lambda'], 
              alpha=0.3, label='Raw data', color='gray', s=20)
    
    # Plot smoothed data
    ax.plot(gdf['dist_out'], gdf[f'smoothed_lambda_{distance_type}'], 
           label='Smoothed data', color='red', linewidth=2)
    
    ax.set_xlabel('Distance from Outlet (km)')
    ax.set_ylabel('λ')
    ax.set_yscale('log')
    ax.set_title(f'{river_name}: Raw vs Smoothed Data')
    ax.legend()
    ax.invert_xaxis()
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize Earth Engine with the updated credentials before processing
    initialize_gee()
    
    logging.basicConfig(level=logging.INFO)
    
    river_names = ["MUSA", "B14_2", "ARG_LAKE_2", "COLOMBIA_2011_2", "VENEZ_2022_N", 
                   "VENEZ_2023", "ANJOBONY", "RUVU_2", "V7_2", "V11_2", "LILONGWE", 
                   "BEMARIVO", "MANGOKY", "TURKWEL"]

    
    # Load characteristic distances from supplementary table
    characteristic_distances = load_characteristic_distances()
    
    results = {}
    for river in river_names:
        if river not in characteristic_distances:
            logging.warning(f"No characteristic distance found for {river}, skipping...")
            continue
            
        logging.info(f"Processing {river}")
        try:
            river_results = analyze_river(river, characteristic_distances[river])
            results[river] = river_results
            
            # Plot results for each distance metric
            for distance_type, result_tuple in river_results.items():
                gdf, lisa_values, p_values, quadrants = result_tuple
                plot_river_analysis(  # This line is where the plotting occurs
                    gdf=gdf,
                    lisa_values=lisa_values,
                    p_values=p_values,
                    quadrants=quadrants,
                    river_name=river,
                    distance_type=distance_type,
                    save_path=f"figures/{distance_type}"
                )
                
        except Exception as e:
            logging.error(f"Error processing river {river}: {str(e)}")
            continue
    
    return results

if __name__ == "__main__":
    results = main()
