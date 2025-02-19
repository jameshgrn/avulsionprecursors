"""Digital Elevation Model operations using Google Earth Engine."""
from typing import Optional
import ee
import geemap
import geopandas as gpd
import numpy as np
from ..sword.base import SwordNode, SwordReach

class DEMSampler:
    """Handles DEM sampling operations."""
    
    def __init__(self, dem_type: str = 'fabdem'):
        """
        Initialize DEM sampler.
        
        Args:
            dem_type: Type of DEM to use ('fabdem' or 'arcticdem')
        """
        self.dem_type = dem_type
        self.dem = self._get_dem()
    
    def _get_dem(self) -> ee.Image:
        """Get the appropriate DEM image."""
        if self.dem_type == 'fabdem':
            dem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
            return dem.mosaic().setDefaultProjection('EPSG:4326', None, 30)
        elif self.dem_type == 'arcticdem':
            dem = ee.Image("UMN/PGC/ArcticDEM/V3/2m_mosaic")
            return dem.setDefaultProjection('EPSG:4326', None, 2)
        else:
            raise ValueError(f"Unsupported DEM type: {self.dem_type}")
    
    def sample_reach(self, reach: SwordReach) -> gpd.GeoDataFrame:
        """
        Sample elevations for all cross-section points in a reach.
        
        Args:
            reach: SwordReach object with cross-sections
            
        Returns:
            GeoDataFrame with sampled elevations
        """
        # Convert cross-section points to EE features
        points = []
        for node in reach.nodes:
            if not node.cross_section:
                continue
            points.extend(self._node_to_features(node))
            
        if not points:
            raise ValueError("No valid cross-section points found")
            
        # Create feature collection and sample
        fc = ee.FeatureCollection(points)
        sampled = self.dem.sampleRegions(
            collection=fc,
            scale=2,
            geometries=True
        )
        
        # Export to GeoDataFrame
        return self._export_to_gdf(sampled, reach.reach_id)
    
    def _node_to_features(self, node: SwordNode) -> list:
        """Convert node cross-section points to EE features."""
        features = []
        if node.cross_section:
            for i, point in enumerate(node.cross_section.coords):
                feature = ee.Feature(
                    ee.Geometry.Point(point),
                    {
                        'node_id': node.node_id,
                        'point_number': i
                    }
                )
                features.append(feature)
        return features
    
    def _export_to_gdf(self, sampled: ee.FeatureCollection, reach_id: int) -> gpd.GeoDataFrame:
        """
        Export sampled features to GeoDataFrame.
        
        Args:
            sampled: EE FeatureCollection with sampled elevations
            reach_id: ID of the reach being processed
            
        Returns:
            GeoDataFrame with elevation data
        """
        # Convert to GeoDataFrame using geemap
        gdf = geemap.ee_to_gdf(sampled)
        
        # Rename elevation column from 'b1' to 'elevation'
        if 'b1' in gdf.columns:
            gdf = gdf.rename(columns={'b1': 'elevation'})
        
        # Ensure required columns exist
        required_cols = {'node_id', 'point_number', 'elevation', 'geometry'}
        missing_cols = required_cols - set(gdf.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add reach_id if not present
        if 'reach_id' not in gdf.columns:
            gdf['reach_id'] = reach_id
        
        # Create unique cross_id
        gdf['cross_id'] = gdf.apply(
            lambda row: f"{row['node_id']}_{row['reach_id']}", 
            axis=1
        )
        
        # Ensure proper CRS
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        
        # Add quality flags
        gdf['elevation_quality'] = gdf['elevation'].apply(
            lambda x: 'valid' if x is not None and not np.isnan(x) else 'invalid'
        )
        
        return gdf 