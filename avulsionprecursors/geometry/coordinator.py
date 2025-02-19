"""Coordinator for geometric operations on river networks."""
from typing import List, Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from ..sword.base import SwordNode, SwordReach
from .operations import calculate_bearing, adjust_slope_per_segment, calculate_mean_direction
from .cross_section_analyzer import CrossSectionAnalyzer

class GeometryCoordinator:
    """Coordinates geometric operations for river analysis."""
    
    def __init__(self, reach: SwordReach):
        self.reach = reach
        self._prepare_nodes()
    
    def _prepare_nodes(self) -> None:
        """Calculate azimuths and create cross-sections for nodes with valid neighbors.
           Skips the first and last node since they lack both a previous and next neighbor."""
        nodes = sorted(self.reach.nodes, key=lambda x: x.dist_out, reverse=True)
        # Only process nodes that have both neighbors (drop the first and last)
        for i in range(1, len(nodes) - 1):
            node = nodes[i]
            prev_node = nodes[i - 1]
            next_node = nodes[i + 1]
            node.calculate_azimuth(prev_node, next_node)
            node.create_cross_section()
    
    def create_cross_section_points(self, spacing: float = 2.0) -> gpd.GeoDataFrame:
        """
        Create points along cross-sections for all nodes.
        
        Args:
            spacing: Distance between points in meters
            
        Returns:
            GeoDataFrame with cross-section points and attributes
        """
        cross_sections = []
        for node in self.reach.nodes:
            if not node.cross_section:
                continue
                
            length = node.cross_section.length
            num_points = int(length / spacing)
            points = [node.cross_section.interpolate(i * spacing) for i in range(num_points + 1)]
            
            # Calculate distances along cross-section
            distances: List[float] = [0.0]
            for i in range(1, len(points)):
                dist = points[i].distance(points[i-1])
                distances.append(float(distances[-1] + dist))
            
            for i, (point, dist) in enumerate(zip(points, distances)):
                cross_sections.append({
                    'node_id': node.node_id,
                    'reach_id': self.reach.reach_id,
                    'dist_out': node.dist_out,
                    'cross_id': f"{node.node_id}_{self.reach.reach_id}",
                    'point_number': i,
                    'dist_along': dist,
                    'geometry': point,
                    'width': node.width,
                    'slope': node.slope,
                    'azimuth': node.azimuth
                })
        
        if not cross_sections:
            # Define the expected columns so that the GeoDataFrame has a geometry column even if empty.
            columns = ['node_id', 'reach_id', 'dist_out', 'cross_id', 'point_number',
                       'dist_along', 'geometry', 'width', 'slope', 'azimuth']
            return gpd.GeoDataFrame(columns=columns, crs='EPSG:3857')
        gdf = gpd.GeoDataFrame(cross_sections, crs='EPSG:3857')
        return self._adjust_slopes(gdf)
    
    def _adjust_slopes(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Adjust slopes based on azimuth differences."""
        if 'slope' not in gdf.columns or 'azimuth' not in gdf.columns:
            return gdf
            
        mean_direction = calculate_mean_direction([node.cross_section 
                                                 for node in self.reach.nodes 
                                                 if node.cross_section])
        
        gdf['adjusted_slope'] = gdf.apply(
            lambda row: adjust_slope_per_segment(
                row['slope'], 
                abs(np.degrees(row['azimuth'] - mean_direction))
            ) if row['slope'] is not None else None,
            axis=1
        )
        
        return gdf 