#%%
import os
import ee
import geemap
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import time
from gcsfs.core import GCSFileSystem
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_gee():
    """
    Initialize Google Earth Engine with service account credentials.

    This function checks if the credentials file exists and initializes the Google Earth Engine
    with the provided service account credentials.

    Raises:
        Exception: If the initialization of Google Earth Engine fails.
    """
    try:
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        credentials_path = os.getenv('GEE_CREDENTIALS_PATH')
        
        # Check if credentials path exists and is valid
        if not credentials_path or not os.path.isfile(str(credentials_path)):
            print(f"Credentials file does not exist at: {credentials_path}")
            return

        credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
        ee.Initialize(credentials)
        print("Google Earth Engine has been initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")

def get_fabdem():
    """
    Retrieve the FABDEM image collection from Google Earth Engine.

    Returns:
        ee.Image: A mosaic of the FABDEM image collection with a default projection.
    """
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    return fabdem.mosaic().setDefaultProjection('EPSG:4326', None, 30)

def get_arcticdem():
    arcticdem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
    return arcticdem.setDefaultProjection('EPSG:4326', None, 2)

def sample_point_elevation(feature, fabdem):
    """
    Sample the elevation of a given feature using the FABDEM image.

    Parameters:
        feature (ee.Feature): The feature for which to sample the elevation.
        fabdem (ee.Image): The FABDEM image to use for sampling.

    Returns:
        ee.Feature: The feature with the sampled elevation property added.
    """
    elevation = fabdem.sample(region=feature.geometry(), scale=2).first().get('b1')
    return feature.set('elevation', elevation).copyProperties(feature)

def process_all_points(fabdem, cross_section_points, unique_id):
    """
    Process all cross-section points to sample elevations and export the results to Cloud Storage.

    Parameters:
        fabdem (ee.Image): The FABDEM image to use for sampling.
        cross_section_points (gpd.GeoDataFrame): The GeoDataFrame containing cross-section points.
        unique_id (str): A unique identifier for the export tasks.
    """
    # Convert the GeoDataFrame to a FeatureCollection
    fc_points = geemap.gdf_to_ee(cross_section_points)

    # Split the points into chunks
    chunk_size = 5000  # Adjust this value based on your data
    chunks = [cross_section_points.iloc[i:i+chunk_size] for i in range(0, len(cross_section_points), chunk_size)]

    tasks = []  # List to hold all tasks

    for i, chunk in enumerate(chunks):
        # Convert the chunk to a FeatureCollection
        fc_chunk = geemap.gdf_to_ee(chunk)

        # Sample the elevation at all points in the current chunk
        profiles = fabdem.sampleRegions(collection=fc_chunk, scale=30, geometries=True)  # Add geometries=True to retain geometry

        # Export the results to Cloud Storage
        task = ee.batch.Export.table.toCloudStorage(
            collection=profiles,
            description=f'{unique_id}_Cross_Section_Sampling_{i}',
            bucket='leveefinders-test',
            fileNamePrefix=f'{unique_id}_Cross_Section_Sampling_{i}',
            fileFormat='CSV'
        )
        task.start()
        tasks.append(task)  # Add the task to the list

    # Now wait for all tasks to complete
    check_tasks_status(tasks)

def check_tasks_status(tasks):
    """
    Check the status of all export tasks and wait for their completion.

    Parameters:
        tasks (list): A list of Google Earth Engine export tasks.
    """
    for task in tasks:
        while task.active():
            print(f'Task {task.id} is still running')
            time.sleep(10)
        print(f'Task {task.id} is completed')

def read_data_from_gcs(bucket_name, file_prefix):
    """
    Read data from Google Cloud Storage and return it as a GeoDataFrame.

    Parameters:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        file_prefix (str): The prefix of the files to read from the bucket.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the concatenated data from the files.
    """
    service_account = os.getenv('GEE_SERVICE_ACCOUNT')
    credentials_path = os.getenv('GEE_CREDENTIALS_PATH')
    credentials = ee.ServiceAccountCredentials(service_account, credentials_path)
    fs = GCSFileSystem('LeveeFinders', token=credentials)
    files = fs.ls(bucket_name)
    dfs = []
    for file in files:
        if file_prefix in file:
            with fs.open(file) as f:
                df = pd.read_csv(f)
                dfs.append(df)
    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)

    # Rename 'b1' column to 'elevation'
    final_df.rename(columns={'b1': 'elevation'}, inplace=True)

    # Convert JSON strings in '.geo' column to shapely geometry objects and set as active geometry
    final_df['geometry'] = final_df['.geo'].apply(lambda x: shape(json.loads(x)))
    final_df = gpd.GeoDataFrame(final_df, geometry='geometry', crs="EPSG:4326")

    # Drop the '.geo', 'x', and 'y' columns as they are no longer needed
    final_df.drop(columns=['.geo', 'x', 'y'], inplace=True)

    return final_df

def perform_cross_section_sampling(cross_section_points, unique_id):
    """
    Perform cross-section sampling by initializing GEE, processing points, and reading results from GCS.

    Parameters:
        cross_section_points (gpd.GeoDataFrame): The GeoDataFrame containing cross-section points.
        unique_id (str): A unique identifier for the export tasks.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the cross-section points with sampled elevations.
    """
    initialize_gee()
    arcticdem = get_arcticdem()
    process_all_points(arcticdem, cross_section_points, unique_id)
    # Read the data from GCS
    cross_section_points_elevations = read_data_from_gcs('leveefinders-test', f'{unique_id}_Cross_Section_Sampling_')
    return cross_section_points_elevations
# %%
