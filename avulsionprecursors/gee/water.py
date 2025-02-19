"""Water mask operations using Google Earth Engine."""
from typing import Optional
import ee
import geemap
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

class WaterMask:
    """Handles water masking operations using JRC dataset."""
    
    def __init__(self, occurrence_threshold: int = 50):
        """
        Initialize water mask.
        
        Args:
            occurrence_threshold: Minimum water occurrence percentage (0-100)
        """
        self.threshold = occurrence_threshold
        self.water_dataset = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    
    def create_mask(self, geometry: ee.Geometry) -> gpd.GeoDataFrame:
        """
        Create water mask for a given geometry.
        
        Args:
            geometry: EE geometry to mask
            
        Returns:
            GeoDataFrame containing water mask polygons
        """
        # Create water mask
        water = self.water_dataset.select('occurrence').gt(self.threshold)
        
        # Convert to vectors
        vectors = water.reduceToVectors(
            scale=30,
            maxPixels=1e10,
            geometryType='polygon',
            geometry=geometry,
            labelProperty='water',
            reducer=ee.Reducer.countEvery()
        )
        
        # Convert to GeoDataFrame
        gdf = geemap.ee_to_gdf(vectors)
        
        # Clean up geometries
        gdf['geometry'] = gdf['geometry'].apply(self._clean_geometry)
        gdf = gdf[gdf['geometry'].notna()]
        
        # Set CRS if needed
        if gdf.crs is None:
            gdf.set_crs('EPSG:4326', inplace=True)
        
        return gdf
    
    def _clean_geometry(self, geom) -> Optional[Polygon]:
        """Clean and validate geometry."""
        if geom is None:
            return None
            
        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
            except:
                return None
                
        if isinstance(geom, MultiPolygon):
            # Take largest polygon
            if not geom.is_empty:
                return max(geom.geoms, key=lambda x: x.area)
            return None
            
        return geom if isinstance(geom, Polygon) else None 