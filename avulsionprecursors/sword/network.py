"""Functions for processing SWORD river networks."""
from typing import Set, List
import geopandas as gpd
from .base import SwordReach, SwordNode

def process_network(
    starting_reach_id: int,
    df: gpd.GeoDataFrame,
    node_gdf: gpd.GeoDataFrame,
    start_dist: float,
    end_dist: float
) -> List[SwordReach]:
    """
    Process a river network starting from a given reach.
    
    Args:
        starting_reach_id: ID of the starting reach
        df: GeoDataFrame containing reach data
        node_gdf: GeoDataFrame containing node data
        start_dist: Starting distance from outlet
        end_dist: Ending distance from outlet
        
    Returns:
        List of processed SwordReach objects
    """
    result = []
    visited = set()
    
    def process_neighbours(current_reach_id: int, direction: str) -> None:
        if current_reach_id in visited:
            return
            
        visited.add(current_reach_id)
        row = df[df['reach_id'] == current_reach_id].iloc[0]
        
        # Get nodes for this reach
        reach_nodes = node_gdf[
            (node_gdf['reach_id'] == current_reach_id) &
            (node_gdf['dist_out'] <= start_dist) &
            (node_gdf['dist_out'] >= end_dist)
        ]
        
        nodes = [SwordNode.from_gdf(row) for _, row in reach_nodes.iterrows()]
        
        reach = SwordReach(
            reach_id=current_reach_id,
            nodes=nodes,
            rch_id_up=[int(x) for x in str(row['rch_id_up']).split()],
            rch_id_dn=[int(x) for x in str(row['rch_id_dn']).split()],
            facc=row['facc'],
            geometry=row['geometry']
        )
        result.append(reach)
        
        # Process upstream/downstream based on direction
        if direction == 'up':
            next_ids = reach.rch_id_up
        else:
            next_ids = reach.rch_id_dn
            
        if len(next_ids) > 1:
            # Get reach with highest flow accumulation
            next_reaches = df[df['reach_id'].isin(next_ids)]
            next_id = next_reaches.loc[next_reaches['facc'].idxmax()]['reach_id']
            process_neighbours(next_id, direction)
        elif len(next_ids) == 1:
            process_neighbours(next_ids[0], direction)
    
    # Process in both directions
    process_neighbours(starting_reach_id, 'up')
    visited.clear()
    process_neighbours(starting_reach_id, 'down')
    
    return result 