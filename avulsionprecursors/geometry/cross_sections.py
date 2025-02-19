"""Functions for creating and analyzing river cross-sections."""
from typing import List, Tuple
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from ..sword.base import SwordNode, SwordReach

def create_cross_section_points(node: SwordNode, spacing: float = 2.0) -> List[Point]:
    """
    Create points along a node's cross-section at regular intervals.
    
    Args:
        node: SwordNode with calculated cross-section
        spacing: Distance between points in meters
        
    Returns:
        List of Points along the cross-section
    """
    if not node.cross_section:
        raise ValueError("Node must have cross_section calculated")
        
    length = node.cross_section.length
    num_points = int(length / spacing)
    return [node.cross_section.interpolate(i * spacing) for i in range(num_points + 1)]

def process_reach_geometry(reach: SwordReach) -> gpd.GeoDataFrame:
    """
    Process all geometric calculations for a reach.
    
    Args:
        reach: SwordReach object containing nodes
        
    Returns:
        GeoDataFrame with cross-section points and attributes
    """
    # Calculate azimuths for all nodes
    nodes = sorted(reach.nodes, key=lambda x: x.dist_out, reverse=True)
    for i, node in enumerate(nodes):
        prev_node = nodes[i-1] if i > 0 else None
        next_node = nodes[i+1] if i < len(nodes)-1 else None
        node.calculate_azimuth(prev_node, next_node)
        node.create_cross_section()
    
    # Create cross-section points
    cross_sections = []
    for node in nodes:
        points = create_cross_section_points(node)
        for i, point in enumerate(points):
            cross_sections.append({
                'node_id': node.node_id,
                'reach_id': reach.reach_id,
                'dist_out': node.dist_out,
                'cross_id': f"{node.node_id}_{node.reach_id}",
                'point_number': i,
                'dist_along': i * 2.0,  # spacing between points
                'geometry': point,
                'width': node.width,
                'slope': node.slope
            })
    
    return gpd.GeoDataFrame(cross_sections, crs='EPSG:3857') 