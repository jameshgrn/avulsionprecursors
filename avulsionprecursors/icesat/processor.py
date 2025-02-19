"""ICESat-2 data processing."""
from typing import Optional, List
import numpy as np
import geopandas as gpd
import sliderule
from pygeodesy.geoids import GeoidKarney
from pygeodesy.points import Numpy2LatLon
import pandas as pd
import pyproj
from ..sword.base import SwordReach
from .config import ICESat2Config
from typing import Dict, Any

class ICESat2Processor:
    """Processes ICESat-2 data for river analysis."""
    
    def __init__(
        self, 
        config: ICESat2Config,
        geoid_path: str = "/Users/jakegearon/geoids/egm2008-1.pgm"
    ):
        self.config = config
        self.geoid = GeoidKarney(geoid_path)
        self.transformer = pyproj.Transformer.from_crs(
            "EPSG:7912", "EPSG:9518", always_xy=True
        )
        
    def initialize(self) -> None:
        """Initialize sliderule connection."""
        sliderule.init(
            self.config.endpoint,
            verbose=False
        )
    
    def process_reach(self, reach: SwordReach) -> gpd.GeoDataFrame:
        """
        Process ICESat-2 data for a reach.
        
        Args:
            reach: SwordReach object to process
            
        Returns:
            GeoDataFrame with processed ICESat-2 data
        """
        # Create buffer around reach for sampling
        buffer_width = max(node.width for node in reach.nodes) * 10
        region = self._create_sampling_region(reach, buffer_width)
        
        # Get ICESat-2 data
        params = {**self.config.to_params(), "poly": region['poly']}
        responses = sliderule.icesat2.atl03sp(params)
        
        if len(responses) == 0:
            raise ValueError(f"No ICESat-2 data found for reach {reach.reach_id}")
        
        # Process responses
        df = gpd.GeoDataFrame(responses)
        return self._process_elevations(df)
    
    def _create_sampling_region(
        self, 
        reach: SwordReach, 
        buffer_width: float
    ) -> Dict[str, Any]:
        """Create sampling region from reach geometry."""
        # Create a GeoDataFrame from the reach geometry in its geographic CRS.
        reach_gdf = gpd.GeoDataFrame(
            geometry=[reach.geometry],
            crs="EPSG:4326"
        )
        # Reproject to a projected CRS (e.g., EPSG:3857) for buffering in meters.
        reach_gdf_proj = reach_gdf.to_crs(epsg=3857)
        # Apply buffer in projected units (meters).
        buffered_proj = reach_gdf_proj.buffer(buffer_width)
        # Convert the buffered geometries back to geographic CRS.
        buffered_geo = buffered_proj.to_crs(epsg=4326)
        # Wrap the buffered geometries into a GeoDataFrame.
        buffered_gdf = gpd.GeoDataFrame(geometry=buffered_geo, crs="EPSG:4326")
        return sliderule.toregion(buffered_gdf, tolerance=0.001)
    
    def _process_elevations(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process elevation data from ICESat-2."""
        # Add unique identifier
        df['UID'] = df.groupby(['rgt', 'cycle', 'spot']).ngroup().add(1)
        
        # Extract coordinates
        df['lat'], df['lon'] = df.geometry.y, df.geometry.x
        
        # Calculate along-track distance using x_atc (as in the older working version)
        df['min_x_atc'] = df.groupby(['rgt', 'cycle', 'spot'])['x_atc'].transform('min')
        df['along_track'] = df['x_atc'] - df['min_x_atc']
        
        # Calculate orthometric heights
        lat_lon_array = np.column_stack((
            df['lat'].values, 
            df['lon'].values, 
            df['height'].values
        ))
        lat_lon_points = Numpy2LatLon(lat_lon_array, ilat=0, ilon=1)
        
        # Get geoid heights
        df['geoid_height'] = [self.geoid(point) for point in lat_lon_points]
        
        # Transform coordinates
        (
            df['transformed_lon'], 
            df['transformed_lat'], 
            df['transformed_z']
        ) = self.transformer.transform(
            lat_lon_array[:, 1], 
            lat_lon_array[:, 0], 
            lat_lon_array[:, 2]
        )
        
        # Calculate orthometric height
        df['orthometric_height'] = df['transformed_z'] - df['geoid_height']
        
        # Debug: print existing columns to verify time field
        print("Process Elevations: DataFrame columns:", df.columns.tolist())

        # Check if the expected 'time' field is present, otherwise try an alternate key
        if 'time' not in df.columns and 'TIME' in df.columns:
            df = df.rename(columns={'TIME': 'time'})

        # Add time information - if still missing, assign NaT
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        else:
            df['time'] = pd.NaT
        df['day'] = df['time'].dt.normalize()
        df['year'] = df['time'].dt.year
        
        return df.to_crs("EPSG:4326")

def process_reach_wse_profile(reach: SwordReach) -> gpd.GeoDataFrame:
    """
    Process the water surface elevation (WSE) profile for the given reach.

    Args:
        reach (SwordReach): The reach object to process.

    Returns:
        GeoDataFrame: Processed ICESat-2 data (WSE profile) for the reach.
    """
    # Instantiate the default configuration
    config = ICESat2Config()
    # Create the processor instance and initialize it
    processor = ICESat2Processor(config)
    processor.initialize()
    # Process and return the WSE profile for the reach
    return processor.process_reach(reach) 