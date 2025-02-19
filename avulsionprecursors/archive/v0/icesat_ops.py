import os
import numpy as np
import pandas as pd
import geopandas as gpd
import ee 
import geemap
import pyproj
import matplotlib.pyplot as plt
from pygeodesy.points import Numpy2LatLon
from pygeodesy.geoids import GeoidKarney
from sqlalchemy import create_engine
import sliderule
from sliderule import icesat2
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from gee_sampler import initialize_gee
from dotenv import load_dotenv
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import time
import logging
import shapely
import math
from scipy.spatial import cKDTree

# Constants and Configuration
WORKING_DIR = '/Users/jakegearon/projects/river_cross_section'
GEOID_PATH = "/Users/jakegearon/geoids/egm2008-1.pgm"
RIVER_NAME = 'ANJOBONY'
INPUT_FILE = f'src/data_handling/data/{RIVER_NAME}_output_based.csv'
OUTPUT_FILE = f'src/data_handling/data/{RIVER_NAME}_output_based_edited_wse.csv'
PLOTS_DIR = 'plots'

# ICESat-2 and GEE parameters
BUFFER_FACTOR = 10
POLY_TOLERANCE = 0.001
# ICESAT2_PARAMS = {
#     "srt": 0,
#     "cnf": 0,
#     "atl08_class": "ground"
# }
ICESAT2_PARAMS = {
    "srt": icesat2.ATL08_WATER,
    "cnf": 3,
    "len": 60,
    "res": 30
}

# Analysis parameters
MIN_SLOPE = 1e-5
MIN_SUPERELEVATION = 0.01
RANSAC_DEGREE = 5
LOWESS_FRAC = 0.2

# Load environment variables
load_dotenv()
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def load_and_preprocess_data(input_file, db_params):
    river_data = pd.read_csv(input_file)
    print(f"Initial river_data shape: {river_data.shape}")

    # Remove duplicates
    river_data = river_data.drop_duplicates(subset=['reach_id', 'node_id'], keep='first')
    print(f"River data shape after removing duplicates: {river_data.shape}")

    # Database connection
    engine = create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

    # Query nodes from the database
    rid_tuple = tuple(int(rid) for rid in river_data['reach_id'].unique())
    sql = f"SELECT * FROM sword_nodes_v16 WHERE reach_id IN {rid_tuple};"
    node_df = gpd.read_postgis(sql, engine, geom_col='geometry', crs='EPSG:4326')

    # Remove duplicates from node_df
    node_df = node_df.drop_duplicates(subset=['reach_id', 'node_id'], keep='first') #type: ignore

    # Merge data and convert to GeoDataFrame
    river_data = river_data.merge(node_df[['reach_id', 'node_id', 'geometry']], on=['reach_id', 'node_id'], how='left')
    river_data['river_name'] = os.path.basename(input_file).split('_output')[0]  # Modified this line

    gdf_proj = gpd.GeoDataFrame(river_data, geometry='geometry', crs='EPSG:4326') #type: ignore
    
    # Remove invalid geometries
    gdf_proj = gdf_proj[gdf_proj.geometry.is_valid]

    # Project to EPSG:3857
    gdf_proj = gdf_proj.to_crs(epsg=3857) #type: ignore

    # Remove geometries with invalid coordinates
    gdf_proj = gdf_proj[~gdf_proj.geometry.apply(lambda geom: any(math.isinf(coord) or math.isnan(coord) for coord in geom.coords[0]))]

    return gdf_proj

def process_atl03_data(df_sr):
    """
    Process ICESat-2 data to add unique identifiers, transform coordinates, and calculate orthometric heights.

    Parameters:
    df_sr (GeoDataFrame): Input GeoDataFrame containing ICESat-2 data with geometry, latitude, longitude, and height information.

    Returns:
    GeoDataFrame: Processed GeoDataFrame with additional columns for transformed coordinates, geoid height, orthometric height, and time-related information.
    """
    # Add unique identifier for each group of rgt, cycle, and spot
    df_sr['UID'] = df_sr.groupby(['rgt', 'cycle', 'spot', 'pair']).ngroup().add(1)
    
    # Extract latitude and longitude from geometry
    df_sr['lat'], df_sr['lon'] = df_sr.geometry.y, df_sr.geometry.x

    min_d = df_sr.segment_dist.min()
    
    df_sr['along_track'] = (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d) - \
                        (((df_sr["segment_dist"] + df_sr["x_atc"])) - min_d).iloc[0] if not df_sr.empty else 0

    # Initialize geoid interpolator and coordinate transformer
    ginterpolator = GeoidKarney(GEOID_PATH)
    transformer = pyproj.Transformer.from_crs("EPSG:7912", "EPSG:9518", always_xy=True)
    
    # Create an array of latitude, longitude, and height
    lat_lon_array = np.column_stack((df_sr['lat'].values, df_sr['lon'].values, df_sr['height'].values))
    
    # Convert array to Numpy2LatLon points
    lat_lon_points = Numpy2LatLon(lat_lon_array, ilat=0, ilon=1)
    
    # Calculate geoid heights for each point
    geoid_heights = [ginterpolator(point) for point in lat_lon_points]
    
    # Transform coordinates to a different CRS
    df_sr['transformed_lon'], df_sr['transformed_lat'], df_sr['transformed_z'] = transformer.transform(lat_lon_array[:, 1], lat_lon_array[:, 0], lat_lon_array[:, 2])
    
    # Add geoid height to the DataFrame
    df_sr['geoid_height'] = np.array(geoid_heights)
    
    # Calculate orthometric height
    df_sr['orthometric_height'] = df_sr['transformed_z'] - df_sr['geoid_height']

    # Reset index and add time-related columns
    icesat_gdf = df_sr.reset_index()
    icesat_gdf['time'] = pd.to_datetime(icesat_gdf['time'])
    icesat_gdf['day'] = icesat_gdf['time'].dt.normalize()
    icesat_gdf['year'] = icesat_gdf['time'].dt.year
    
    # Transform GeoDataFrame to EPSG:4326 CRS
    icesat_gdf = icesat_gdf.to_crs("EPSG:4326")

    return icesat_gdf

def fetch_icesat2_data(gdf_proj):
    polygon4326 = gdf_proj.buffer(gdf_proj['width'].mean() * BUFFER_FACTOR).to_crs("EPSG:4326").union_all()
    polygon_gdf = gpd.GeoDataFrame(geometry=[polygon4326], crs="EPSG:4326") #type: ignore
    if polygon4326.geom_type == 'Polygon':
        ee_polygon = ee.Geometry.Polygon(list(polygon4326.exterior.coords)) #type: ignore
    elif polygon4326.geom_type == 'MultiPolygon':
        ee_polygon = ee.Geometry.MultiPolygon([list(poly.exterior.coords) for poly in polygon4326.geoms]) #type: ignore 
    else:
        raise ValueError(f"Unexpected geometry type: {polygon4326.geom_type}")
    region4326 = sliderule.toregion(polygon_gdf, tolerance=POLY_TOLERANCE)

    parms = {**ICESAT2_PARAMS, "poly": region4326['poly']}
    responses = sliderule.icesat2.atl06p(parms)
    print("Returned keys:", list(responses[0].keys()) if responses else "No responses")
    df_sr = gpd.GeoDataFrame(responses)
    if df_sr.empty:
        raise RuntimeError("No results returned")
    df_sr = process_atl06_data(df_sr)
    
    return df_sr, ee_polygon



def process_atl06_data(df_sr):
    df_sr['UID'] = df_sr.groupby(['rgt', 'cycle', 'spot']).ngroup().add(1)
    df_sr['lat'], df_sr['lon'] = df_sr.geometry.y, df_sr.geometry.x
    df_sr['min_x_atc'] = df_sr.groupby(['rgt', 'cycle', 'spot'])['x_atc'].transform('min')
    df_sr['along_track'] = df_sr['x_atc'] - df_sr['min_x_atc']

    ginterpolator = GeoidKarney(GEOID_PATH)
    transformer = pyproj.Transformer.from_crs("EPSG:7912", "EPSG:9518", always_xy=True)
    lat_lon_array = np.column_stack((df_sr['lat'].values, df_sr['lon'].values, df_sr['h_mean'].values))
    lat_lon_points = Numpy2LatLon(lat_lon_array, ilat=0, ilon=1)
    geoid_heights = [ginterpolator(point) for point in lat_lon_points]
    df_sr['transformed_lon'], df_sr['transformed_lat'], df_sr['transformed_z'] = transformer.transform(lat_lon_array[:, 1], lat_lon_array[:, 0], lat_lon_array[:, 2])
    df_sr['geoid_height'] = np.array(geoid_heights)
    df_sr['orthometric_height'] = df_sr['transformed_z'] - df_sr['geoid_height']

    icesat_gdf = df_sr.reset_index()
    icesat_gdf['time'] = pd.to_datetime(icesat_gdf['time'])
    icesat_gdf['day'] = icesat_gdf['time'].dt.normalize()
    icesat_gdf['year'] = icesat_gdf['time'].dt.year
    icesat_gdf = icesat_gdf.to_crs("EPSG:4326")

    return icesat_gdf

def create_water_mask(ee_polygon, icesat_gdf):
    dataset = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") #type: ignore
    water = dataset.select('occurrence').gt(50)
    vector_water = water.reduceToVectors(scale=30, maxPixels=1e10, geometryType='polygon', geometry=ee_polygon, labelProperty='water', reducer=ee.Reducer.countEvery()) #type: ignore
    vector_water_gdf = geemap.ee_to_gdf(vector_water)

    icesat_gdf = icesat_gdf.to_crs(epsg=3857)
    vector_water_gdf = vector_water_gdf.to_crs(epsg=3857)
    vector_water_gdf_valid = gpd.GeoDataFrame(geometry=vector_water_gdf.make_valid()) #type: ignore

    icesat_gdf_within_polygons = gpd.sjoin(icesat_gdf, vector_water_gdf_valid, how="inner", predicate="intersects")

    return icesat_gdf_within_polygons

def find_median_elevation(node, points_gdf):
    points_gdf = points_gdf.to_crs(3857)
    node = shapely.geometry.Point(node.x, node.y)
    
    points_gdf['distance'] = points_gdf.geometry.distance(node)
    points_within_250m = points_gdf[points_gdf['distance'] <= 250]
    
    if points_within_250m.empty:
        return np.nan
    
    closest_points = points_within_250m.nsmallest(5, 'distance')
    return closest_points['orthometric_height'].median()

def apply_ransac_regression(cleaned_df):
    X = cleaned_df['dist_out'].values.reshape(-1, 1)
    y = cleaned_df['wse_is2'].values
    
    if len(X) < 2:
        print("Warning: Not enough data points for RANSAC regression. Returning original data.")
        return X, y, X, y

    poly = PolynomialFeatures(degree=RANSAC_DEGREE, include_bias=False)
    ransac = RANSACRegressor(random_state=42, min_samples=2)
    model = make_pipeline(poly, ransac)
    
    try:
        model.fit(X, y)
        X_pred = np.linspace(X.min(), X.max(), len(X)).reshape(-1, 1)
        y_pred = model.predict(X_pred)
    except ValueError as e:
        print(f"RANSAC regression failed: {e}")
        print("Falling back to simple linear regression.")
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        X_pred = X
        y_pred = model.predict(X_pred)

    return X, y, X_pred, y_pred

def interpolate_lowess(x, y, frac=LOWESS_FRAC, x_pred=None):
    lowess_results = lowess(y, x, frac=frac)
    x_lowess, y_lowess = lowess_results[:, 0], lowess_results[:, 1]
    lowess_interp = interp1d(x_lowess, y_lowess, bounds_error=False, fill_value="extrapolate") #type: ignore
    if x_pred is None:
        x_pred = x
    y_pred = lowess_interp(x_pred)
    return y_pred

def calculate_river_parameters(gdf_proj):
    gdf_proj['interpolated_wse'] = interpolate_lowess(gdf_proj['dist_out'].values, gdf_proj['wse_is2'].values)
    gdf_proj['lower_ridge'] = gdf_proj[['ridge1_elevation', 'ridge2_elevation']].min(axis=1)
    gdf_proj['interpolated_wse'] = np.minimum(gdf_proj['interpolated_wse'], gdf_proj['lower_ridge'])

    epsilon = 1e-10

    # Apply minimum threshold to slope
    gdf_proj['slope'] = np.maximum(gdf_proj['slope'], MIN_SLOPE)

    # Calculate slopes
    gdf_proj['ridge1_slope'] = (gdf_proj['ridge1_elevation'] - gdf_proj['floodplain1_elevation']) / (abs(gdf_proj['ridge1_dist_along'] - gdf_proj['floodplain1_dist_along']) + epsilon)
    gdf_proj['ridge2_slope'] = (gdf_proj['ridge2_elevation'] - gdf_proj['floodplain2_elevation']) / (abs(gdf_proj['ridge2_dist_along'] - gdf_proj['floodplain2_dist_along']) + epsilon)

    # Handle NaNs in slope calculations
    gdf_proj['ridge1_slope'] = gdf_proj['ridge1_slope'].replace([np.inf, -np.inf], np.nan)
    gdf_proj['ridge2_slope'] = gdf_proj['ridge2_slope'].replace([np.inf, -np.inf], np.nan)

    # Calculate gamma values
    gdf_proj['gamma1'] = np.abs(gdf_proj['ridge1_slope']) / gdf_proj['slope']
    gdf_proj['gamma2'] = np.abs(gdf_proj['ridge2_slope']) / gdf_proj['slope']

    # Handle NaNs in gamma calculations
    gdf_proj['gamma1'] = gdf_proj['gamma1'].replace([np.inf, -np.inf], np.nan)
    gdf_proj['gamma2'] = gdf_proj['gamma2'].replace([np.inf, -np.inf], np.nan)

    gdf_proj['gamma_mean'] = gdf_proj[['gamma1', 'gamma2']].mean(axis=1, skipna=True)

    # Calculate a_b and corrected_denominator
    gdf_proj['a_b_1'] = (gdf_proj['ridge1_elevation'] - gdf_proj['interpolated_wse']) / (gdf_proj['XGB_depth'] + epsilon)
    gdf_proj['a_b_2'] = (gdf_proj['ridge2_elevation'] - gdf_proj['interpolated_wse']) / (gdf_proj['XGB_depth'] + epsilon)
    gdf_proj['a_b'] = gdf_proj[['a_b_1', 'a_b_2']].mean(axis=1, skipna=True)

    conditions = [gdf_proj['a_b'] <= 1.25, gdf_proj['a_b'] > 1.25]
    choices = [
        gdf_proj['XGB_depth'],
        gdf_proj[['ridge1_elevation', 'ridge2_elevation']].mean(axis=1, skipna=True) - gdf_proj['interpolated_wse']
    ]
    gdf_proj['corrected_denominator'] = np.select(conditions, choices)

    # Calculate superelevation
    gdf_proj['superelevation1'] = np.maximum((gdf_proj['ridge1_elevation'] - gdf_proj['floodplain1_elevation']) / (gdf_proj['corrected_denominator'] + epsilon), MIN_SUPERELEVATION)
    gdf_proj['superelevation2'] = np.maximum((gdf_proj['ridge2_elevation'] - gdf_proj['floodplain2_elevation']) / (gdf_proj['corrected_denominator'] + epsilon), MIN_SUPERELEVATION)
    gdf_proj['superelevation_mean'] = gdf_proj[['superelevation1', 'superelevation2']].mean(axis=1, skipna=True)

    # Calculate lambda and add single-side measurement flag
    gdf_proj['lambda'] = gdf_proj['gamma_mean'] * gdf_proj['superelevation_mean']
    gdf_proj['single_side_measurement'] = gdf_proj['gamma2'].isna()

    # Track NaN values in lambda
    nan_lambda_count = gdf_proj['lambda'].isna().sum()
    print(f"Number of NaN lambda values: {nan_lambda_count}")

    return gdf_proj

def analyze_elevation_distribution(gdf_proj, icesat_gdf_within_polygons):
    gdf_proj['has_elevation'] = ~gdf_proj['median_ortho_height'].isna()

    fig, ax = plt.subplots(figsize=(12, 8))
    gdf_proj[gdf_proj['has_elevation']].plot(ax=ax, color='blue', label='Valid Elevation', markersize=5)
    gdf_proj[~gdf_proj['has_elevation']].plot(ax=ax, color='red', label='NaN Elevation', markersize=5)
    ax.legend()
    plt.title('Spatial Distribution of Nodes with Valid and NaN Elevations')
    plt.savefig(f'{PLOTS_DIR}/{RIVER_NAME}_elevation_distribution.png')
    plt.close()

    icesat_coords = np.array(list(icesat_gdf_within_polygons.geometry.apply(lambda geom: (geom.x, geom.y))))
    node_coords = np.array(list(gdf_proj.geometry.apply(lambda geom: (geom.x, geom.y))))

    tree = cKDTree(icesat_coords)
    distances, _ = tree.query(node_coords)

    gdf_proj['distance_to_nearest_icesat'] = distances

    print("\nDistance to nearest ICESat-2 point statistics:")
    print(gdf_proj['distance_to_nearest_icesat'].describe())

    print("\nCharacteristics of nodes with valid elevations:")
    print(gdf_proj[gdf_proj['has_elevation']].describe())

    print("\nCharacteristics of nodes with NaN elevations:")
    print(gdf_proj[~gdf_proj['has_elevation']].describe())

    print("\nDistribution of valid/NaN elevations by reach_id:")
    print(gdf_proj.groupby('reach_id')['has_elevation'].value_counts(normalize=True).unstack())

    print("\nPercentage of problematic measurements:")
    print(f"High gamma (>100): {gdf_proj['gamma_mean'].gt(100).mean()*100:.2f}%")
    print(f"Low superelevation (<0.05): {gdf_proj['superelevation_mean'].lt(0.05).mean()*100:.2f}%")
    print(f"Single-side measurements: {gdf_proj['single_side_measurement'].mean()*100:.2f}%")

def main():
    # Set working directory and initialize
    os.chdir(WORKING_DIR)
    icesat2.init("slideruleearth.io", loglevel=logging.DEBUG, verbose=False)
    initialize_gee()

    # Process data
    start_time = time.time()
    gdf_proj = load_and_preprocess_data(INPUT_FILE, DB_PARAMS)
    df_sr, ee_polygon = fetch_icesat2_data(gdf_proj)
    icesat_gdf = process_atl06_data(df_sr)
    icesat_gdf_within_polygons = create_water_mask(ee_polygon, icesat_gdf)

    gdf_proj['median_ortho_height'] = gdf_proj.apply(lambda row: find_median_elevation(row.geometry, icesat_gdf_within_polygons), axis=1)

    gdf_proj['median_ortho_height'] = pd.to_numeric(gdf_proj['median_ortho_height'], errors='coerce')
    gdf_proj['dist_out'] = pd.to_numeric(gdf_proj['dist_out'], errors='coerce')
    gdf_proj['wse_is2'] = pd.to_numeric(gdf_proj['median_ortho_height'], errors='coerce')

    cleaned_df = gdf_proj.dropna(subset=['dist_out', 'wse_is2'])

    if len(cleaned_df) < 2:
        print("Warning: Not enough valid data points after cleaning. Skipping regression and plotting.")
        gdf_proj['interpolated_wse'] = gdf_proj['wse_is2']
    else:
        X, y, X_pred, y_pred = apply_ransac_regression(cleaned_df)

        fig, ax = plt.subplots()
        ax.plot(X, y, 'o', label='Original data', markersize=2)
        ax.plot(X_pred, y_pred, 'r-', label='Fitted line')
        ax.legend()
        plt.savefig(f"{PLOTS_DIR}/{RIVER_NAME}_regression_fit.png")
        plt.close()

        # Interpolate the predicted values back to the original dataframe
        interp_func = interp1d(X_pred.flatten(), y_pred, kind='linear', fill_value='extrapolate') #type: ignore
        gdf_proj['interpolated_wse'] = interp_func(gdf_proj['dist_out'])

    gdf_proj = calculate_river_parameters(gdf_proj)

    gdf_proj.to_csv(OUTPUT_FILE, index=False)
    print(f"Updated data saved to '{OUTPUT_FILE}'")

    analyze_elevation_distribution(gdf_proj, icesat_gdf_within_polygons)

    # Add histograms
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.hist(gdf_proj['slope'], bins=50)
    plt.title('Slope Distribution')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    
    plt.subplot(132)
    plt.hist(gdf_proj['superelevation_mean'], bins=50)
    plt.title('Superelevation Distribution')
    plt.xlabel('Superelevation')
    plt.ylabel('Frequency')
    
    plt.subplot(133)
    plt.hist(gdf_proj['gamma_mean'], bins=50)
    plt.title('Gamma Distribution')
    plt.xlabel('Gamma')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{RIVER_NAME}_distributions.png')
    plt.close()

    # Separate analysis for single-side and double-side measurements
    single_side = gdf_proj[gdf_proj['single_side_measurement']]
    double_side = gdf_proj[~gdf_proj['single_side_measurement']]

    print("\nSingle-side measurements statistics:")
    print(single_side[['gamma_mean', 'superelevation_mean', 'lambda']].describe())

    print("\nDouble-side measurements statistics:")
    print(double_side[['gamma_mean', 'superelevation_mean', 'lambda']].describe())

    print(f"Finished processing {gdf_proj['river_name'].iloc[0]} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
