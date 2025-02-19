import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib
matplotlib.use('Qt5Agg')
from sqlalchemy import create_engine
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
from shapely import wkb
import numpy as np
from shapely.geometry import Point
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import csv
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from PyQt5.QtCore import QThreadPool
from avulsionprecursors.archive.v0.labeling_worker import InferenceWorker

# VENEZ_2023_W last processed index: 386

name = "Kanektok"
n = 1  # adjust this value to skip cross sections


# Load the data
sword_data_gdf = gpd.read_parquet(f'src/data_handling/data/all_elevations_gdf_{name}.parquet')

sword_data_gdf.crs = 'EPSG:4326'

#%%
# Organize data
def organize_data_by_reach_and_node(data_gdf):
    grouped = data_gdf.groupby(['cross_id'])
    # Order from upstream to downstream by using negative dist_out
    cross_sections = sorted([group[1] for group in grouped], key = lambda x: x['dist_out'].mean(),
                            reverse = True)
    return cross_sections


cross_sections = organize_data_by_reach_and_node(sword_data_gdf)


# Define a consistent set of features to use
FEATURES = [
    # Original features
    'width', 'width_var', 'sinuosity', 'max_width', 'dist_out', 
    'n_chan_mod', 'n_chan_max', 'facc', 'meand_len', 'slope',
    
    # Elevation-related features
    'elevation_min', 'elevation_max', 'elevation_range', 'elevation_mean',
    'relative_elevation_mean', 'relative_elevation_max',
    
    # Slope-related features
    'slope_r_mean', 'slope_r_max', 'local_slope_mean', 'local_slope_max',
    
    # Curvature-related features
    'curvature_mean', 'curvature_max', 'local_curvature_mean', 'local_curvature_max',
    
    # Distance-related features
    'dist_along_range',
]

def compute_attributes(df):
    df = df.copy()
    df.set_index('dist_along', inplace=True, drop=False)

    # Compute basic attributes
    df['slope_r'] = df['elevation'].diff().abs() / df.index.to_series().diff()
    df['curvature'] = df['slope_r'].diff() / df.index.to_series().diff()
    df['local_slope'] = df['elevation'].diff() / df['dist_along'].diff()
    df['local_curvature'] = df['local_slope'].diff() / df['dist_along'].diff()
    df['relative_elevation'] = df['elevation'] - df['elevation'].min()

    # Compute summary statistics for these attributes
    attributes = {}
    for feature in FEATURES:
        if feature in df.columns:
            attributes[feature] = df[feature].iloc[0]
        elif feature.endswith('_min'):
            attributes[feature] = df[feature[:-4]].min()
        elif feature.endswith('_max'):
            attributes[feature] = df[feature[:-4]].max()
        elif feature.endswith('_mean'):
            attributes[feature] = df[feature[:-5]].mean()
        elif feature == 'elevation_range':
            attributes[feature] = df['elevation'].max() - df['elevation'].min()
        elif feature == 'dist_along_range':
            attributes[feature] = df['dist_along'].max() - df['dist_along'].min()
        else:
            raise ValueError(f"Feature {feature} could not be computed")

    # Add computed attributes
    attributes.update({
        'min_dist_along': df['dist_along'].min(),
        'max_dist_along': df['dist_along'].max(),
    })
    
    return attributes


def get_empty_dataframe_with_columns():
    position_dependent_attrs = ['elevation', 'slope_r', 'curvature', 'dist_along']
    global_attributes = ['width', 'width_var', 'sinuosity', 'max_width', 'dist_out', 'n_chan_mod', 'n_chan_max', 'facc',
                         'meand_len', 'type', 'reach_id', 'node_id', 'slope']
    labels = ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']

    columns = []
    for label in labels:
        for attr in position_dependent_attrs:
            columns.append(f"{label}_{attr}")
    columns.extend(global_attributes)
    return pd.DataFrame(columns = columns)


def update_and_save_to_csv(df, labeled_points, filename=f"{name}_output.csv"):
    # Start with an empty DataFrame with all the required columns
    attr_df = pd.DataFrame(columns=FEATURES + ['channel_dist_along', 'ridge1_dist_along', 'floodplain1_dist_along', 'ridge2_dist_along', 'floodplain2_dist_along', 'channel_elevation', 'ridge1_elevation', 'floodplain1_elevation', 'ridge2_elevation', 'floodplain2_elevation', 'reach_id', 'node_id'])
    
    if any(v for v in labeled_points.values()):  # Check if labeled_points is not empty
        # Compute the attributes as normal if labeled_points are provided
        attributes = compute_attributes(df)
        
        # Update the values in attr_df for all features
        for feature in FEATURES:
            attr_df.at[0, feature] = attributes.get(feature, np.nan)
        
        # Add reach_id and node_id
        attr_df.at[0, 'reach_id'] = df['reach_id'].iloc[0]
        attr_df.at[0, 'node_id'] = df['node_id'].iloc[0]
    
    # Calculate dist_along and elevation for each labeled point
    for label in ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']:
        if label in labeled_points and labeled_points[label]:
            dist_along = labeled_points[label][0][0]
            attr_df.at[0, f'{label}_dist_along'] = dist_along
            
            # Interpolate elevation for the labeled point
            elevation = np.interp(dist_along, df['dist_along'], df['elevation'])
            attr_df.at[0, f'{label}_elevation'] = elevation
    
    # Check if the file exists, if not, create one with a header
    if not os.path.exists(filename):
        attr_df.to_csv(filename, index=False)
    else:
        # Append to the existing CSV
        with open(filename, 'a', newline='') as f:  # Open the file in append mode
            attr_df.to_csv(f, header=False, index=False)


def get_last_processed_index(river_name, filename="last_processed_{}.txt"):
    if os.path.exists(filename.format(river_name)):
        with open(filename.format(river_name), "r") as f:
            return int(f.read().strip())
    return 0

def save_last_processed_index(idx, river_name, filename="last_processed_{}.txt"):
    with open(filename.format(river_name), "w") as f:
        f.write(str(idx + 1))

# New function to train the model
def train_predictive_model(csv_file, river_name):
    print("Training predictive model...")
    
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. No data to train on.")
        return None, None, None
    
    data = pd.read_csv(csv_file)
    
    if len(data) < 10:
        print("Not enough samples to train the model. Need at least 10 samples.")
        return None, None, None
    
    print("Columns in the CSV:", data.columns.tolist())
    
    # Use only the predefined features (excluding reach_id and node_id)
    X = data[FEATURES]
    target_cols = ['channel_dist_along', 'ridge1_dist_along', 'floodplain1_dist_along', 'ridge2_dist_along', 'floodplain2_dist_along']
    y = data[target_cols]
    
    # Handle NaN values
    valid_rows = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X[valid_rows]
    y = y[valid_rows]
    
    print(f"\nRemoved {(~valid_rows).sum()} rows with NaN values")
    print(f"Remaining samples: {len(X)}")
    
    if len(X) == 0:
        print("No valid samples remaining after removing NaN values")
        return None, None, None
    
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_scaler = RobustScaler()
    y_scaled = y_scaler.fit_transform(y)
    
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_scaled)
    
    joblib.dump(model, f'predictive_model_{river_name}.joblib')
    joblib.dump(scaler, f'feature_scaler_{river_name}.joblib')
    joblib.dump(y_scaler, f'y_scaler_{river_name}.joblib')
    print("Model trained and saved.")
    
    return model, scaler, y_scaler

# New function to predict points
def predict_points(df, model, scaler, y_scaler):
    print("Predicting points...")
    
    # Compute attributes including the new derived ones
    attributes = compute_attributes(df)
    
    # Use only the predefined features
    X = pd.DataFrame([attributes])[FEATURES]
    
    print(X.columns)
    
    X_scaled = scaler.transform(X)
    
    try:
        predicted_scaled = model.predict(X_scaled)[0]
        predicted_actual = y_scaler.inverse_transform(predicted_scaled.reshape(1, -1))[0]
        
        min_dist = attributes['min_dist_along']
        max_dist = attributes['max_dist_along']
        
        predicted_actual = np.clip(predicted_actual, min_dist, max_dist)
        
        print("Cross-section range:", min_dist, "to", max_dist)
        print("Predicted points (before clipping):", predicted_scaled)
        print("Predicted points (after clipping):", predicted_actual)
        
        target_cols = ['channel_dist_along', 'ridge1_dist_along', 'floodplain1_dist_along', 'ridge2_dist_along', 'floodplain2_dist_along']
        return dict(zip(target_cols, predicted_actual))
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {}

def label_cross_section(df, model, scaler, y_scaler, current_index, total_cross_sections):
    # Instead of calling plt.show(), create the figure programmatically
    # and leave it to the calling GUI code to display it.
    fig, ax1 = plt.subplots()
    
    # You may remove or comment out plt.show() to avoid blocking.
    # plt.show()  # DO NOT call this here in a Qt application
    
    # Submit the inference task as a background job
    worker = InferenceWorker(df, model, scaler, y_scaler)
    
    def handle_finished(predicted_points):
        print("Predicted points received:", predicted_points)
        colors = {
            'channel': 'blue',
            'ridge1': 'green',
            'floodplain1': 'red',
            'ridge2': 'green',
            'floodplain2': 'red'
        }
        try:
            # Plot the predicted points directly onto the figure's axes.
            for label, dist in predicted_points.items():
                # Use your own interpolation logic as needed.
                if 'dist_along' in df.columns:
                    y = np.interp(dist, df['dist_along'], df['elevation'])
                else:
                    y = np.interp(dist, df.index, df['elevation'])
                ax1.plot(dist, y, 'X', markersize=10, color=colors[label], alpha=0.5)
            # Update title to include progress information.
            progress = (current_index + 1) / total_cross_sections
            progress_bar = '█' * int(20 * progress) + '-' * (20 - int(20 * progress))
            ax1.set_title(f"Progress: [{progress_bar}] {progress:.1%}")
            # Finally, redraw; if you embed the figure in a Qt widget, trigger its update.
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error while plotting predicted points: {e}")
    
    def handle_error(err_str):
        print("Inference worker error:", err_str)
    
    worker.signals.finished.connect(handle_finished)
    worker.signals.error.connect(handle_error)
    
    QThreadPool.globalInstance().start(worker)

def count_rows_in_csv(csv_file):
    if not os.path.exists(csv_file):
        return 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        row_count = sum(1 for row in reader)
    
    return row_count

def model_exists(river_name):
    return os.path.exists(f'predictive_model_{river_name}.joblib') and \
           os.path.exists(f'feature_scaler_{river_name}.joblib') and \
           os.path.exists(f'y_scaler_{river_name}.joblib')


if __name__ == "__main__":
    print("Script started.")
    # Define n
    labeled_data = []

    print(f"Processing river: {name}")
    total_cross_sections_to_process = len(cross_sections[::n])
    print(f"Total cross sections to process: {total_cross_sections_to_process}")

    # Get the last processed index
    start_idx = get_last_processed_index(name)
    print(f"Starting from index: {start_idx}")

    # Initialize a counter for valid samples
    valid_sample_count = 0

    # Initialize a variable to keep track of the number of labeled data points
    csv_file = f"data/{name}_output.csv"
    labeled_data_points = count_rows_in_csv(csv_file)
    print(f"Existing labeled data points: {labeled_data_points}")

    # Initialize model, scaler, and y_scaler
    model = None
    scaler = None
    y_scaler = None

    # Check if the model exists, and retrain if necessary
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. Starting with no trained model.")
    else:
        print("Attempting to train model from existing data...")
        model, scaler, y_scaler = train_predictive_model(csv_file, name)

    print(f"About to start processing {len(cross_sections[start_idx::n])} cross sections.")

    # Modify the main processing loop
    for idx in range(start_idx, len(cross_sections), n):
        df = cross_sections[idx]
        print(f"Processing cross-section {idx + 1} of {len(cross_sections)}")
        
        remaining = total_cross_sections_to_process - (idx // n) - 1
        print(f"Processing cross-section {idx + 1} of {len(cross_sections)} ({remaining} remaining)")

        label_cross_section(df, model, scaler, y_scaler, idx // n, total_cross_sections_to_process)

        # If no points are labeled and it's the first cross-section, save the empty dataframe
        if idx == 0 and all(not v for v in labeled_points.values()):
            empty_df = pd.DataFrame(columns=FEATURES + ['channel_dist_along', 'ridge1_dist_along', 'floodplain1_dist_along', 'ridge2_dist_along', 'floodplain2_dist_along', 'channel_elevation', 'ridge1_elevation', 'floodplain1_elevation', 'ridge2_elevation', 'floodplain2_elevation', 'reach_id', 'node_id'])
            empty_df.to_csv(csv_file, index=False)
        else:
            labeled_data.append((df, labeled_points))
            # Update and save to CSV after every cross section
            update_and_save_to_csv(df, labeled_points, filename=csv_file)
            save_last_processed_index(idx, name)

            # Increment valid sample count based on labeled points
            valid_sample_count += sum(len(positions) for positions in labeled_points.values())

            # Increment the number of labeled data points
            labeled_data_points += sum(bool(v) for v in labeled_points.values())

        # Check if we have enough samples to retrain
        if labeled_data_points >= 10:
            print("Retraining model...")
            model, scaler, y_scaler = train_predictive_model(csv_file, name)
            labeled_data_points = 0  # Reset the counter after retraining

    print("Script completed.")