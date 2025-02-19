"""Analysis tools for river cross-sections."""
from typing import List, Dict, Any
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from .operations import calculate_bearing, adjust_slope_per_segment
from ..sword.base import SwordNode

class CrossSectionAnalyzer:
    """Analyzer for river cross-sections."""
    
    def __init__(self, node: SwordNode):
        self.node = node
        self.points: List[Point] = []
        self.distances: List[float] = []
        self.elevations: List[float] = []
        
    def add_elevation_point(self, point: Point, elevation: float) -> None:
        """Add an elevation measurement point."""
        self.points.append(point)
        if self.points:
            dist = point.distance(self.points[0])
            self.distances.append(dist)
        else:
            self.distances.append(0.0)
        self.elevations.append(elevation)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate statistical properties of the cross-section."""
        if not self.elevations:
            return {}
            
        center_third_mask = np.logical_and(
            np.array(self.distances) >= np.quantile(self.distances, 0.33),
            np.array(self.distances) <= np.quantile(self.distances, 0.66)
        )
        
        center_elevations = np.array(self.elevations)[center_third_mask]
        outer_elevations = np.array(self.elevations)[~center_third_mask]
        
        return {
            'relief': (np.quantile(center_elevations, 0.95) - 
                      np.quantile(outer_elevations, 0.15)
                      if len(center_elevations) and len(outer_elevations) else None),
            'mean_elevation': np.mean(self.elevations),
            'std_elevation': np.std(self.elevations),
            'skew_elevation': float(np.nanmean(((np.array(self.elevations) - 
                                               np.mean(self.elevations)) / 
                                               np.std(self.elevations)) ** 3))
            if len(self.elevations) > 2 else None
        } 