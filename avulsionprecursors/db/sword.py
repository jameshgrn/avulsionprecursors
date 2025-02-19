"""SWORD database operations."""
from typing import List, Optional, Dict, Any
import geopandas as gpd
from sqlalchemy import create_engine, text
from ..sword.base import SwordNode, SwordReach
from .config import DBConfig

class SWORDDatabase:
    """Interface for SWORD database operations."""
    
    def __init__(self, config: DBConfig):
        """Initialize database connection."""
        self.engine = create_engine(config.connection_string)
        
    def get_reach(self, reach_id: int) -> Optional[SwordReach]:
        """
        Get a single reach and its nodes from the database.
        
        Args:
            reach_id: SWORD reach ID
            
        Returns:
            SwordReach object if found, None otherwise
        """
        # Get reach data
        reach_sql = """
        SELECT * FROM sword_reaches_v16 WHERE reach_id = %(reach_id)s;
        """
        reach_df = gpd.read_postgis(
            reach_sql, 
            self.engine, 
            params={'reach_id': reach_id},
            geom_col='geometry'
        )
        
        if reach_df.empty:
            return None
            
        # Get nodes for this reach
        nodes_sql = """
        SELECT * FROM sword_nodes_v16 
        WHERE reach_id = %(reach_id)s 
        ORDER BY dist_out DESC;
        """
        nodes_df = gpd.read_postgis(
            nodes_sql,
            self.engine,
            params={'reach_id': reach_id},
            geom_col='geometry'
        )
        
        # Convert to SwordNode objects
        nodes = [SwordNode.from_gdf(row) for _, row in nodes_df.iterrows()]
        
        # Create SwordReach
        row = reach_df.iloc[0]
        return SwordReach(
            reach_id=reach_id,
            nodes=nodes,
            rch_id_up=[int(x) for x in str(row['rch_id_up']).split() if x],
            rch_id_dn=[int(x) for x in str(row['rch_id_dn']).split() if x],
            facc=float(row['facc']),
            geometry=row['geometry']
        )
    
    def get_river_reaches(self, river_name: str) -> List[SwordReach]:
        """
        Get all reaches for a given river.
        
        Args:
            river_name: Name of the river
            
        Returns:
            List of SwordReach objects
        """
        # First get all reach IDs for this river
        reach_ids_sql = """
        SELECT DISTINCT reach_id 
        FROM sword_reaches_v16 
        WHERE river_name = :river_name;
        """
        with self.engine.connect() as conn:
            reach_ids = [
                row[0] for row in conn.execute(
                    text(reach_ids_sql), 
                    {'river_name': river_name}
                )
            ]
        
        # Get each reach
        reaches = []
        for rid in reach_ids:
            reach = self.get_reach(rid)
            if reach:
                reaches.append(reach)
        
        return reaches
    
    def get_upstream_reaches(self, reach_id: int, limit: Optional[int] = None) -> List[SwordReach]:
        """Get upstream reaches."""
        reaches = []
        visited = set()
        
        def traverse_upstream(current_id: int, count: int = 0) -> None:
            if current_id in visited or (limit and count >= limit):
                return
                
            reach = self.get_reach(current_id)
            if not reach:
                return
                
            visited.add(current_id)
            reaches.append(reach)
            
            if reach.rch_id_up:
                # Follow the branch with highest flow accumulation
                next_id = max(
                    reach.rch_id_up,
                    key=lambda x: self.get_reach(x).facc if self.get_reach(x) else 0
                )
                traverse_upstream(next_id, count + 1)
        
        traverse_upstream(reach_id)
        return reaches
    
    def get_downstream_reaches(self, reach_id: int, limit: Optional[int] = None) -> List[SwordReach]:
        """
        Get downstream reaches following highest flow accumulation.
        
        Args:
            reach_id: Starting reach ID
            limit: Maximum number of reaches to return
        """
        reaches = []
        visited = set()
        
        def traverse_downstream(current_id: int, count: int = 0) -> None:
            if current_id in visited or (limit and count >= limit):
                return
            
            reach = self.get_reach(current_id)
            if not reach:
                return
            
            visited.add(current_id)
            reaches.append(reach)
            
            if reach.rch_id_dn:
                next_id = max(
                    reach.rch_id_dn,
                    key=lambda x: self.get_reach(x).facc if self.get_reach(x) else 0
                )
                traverse_downstream(next_id, count + 1)
        
        traverse_downstream(reach_id)
        return reaches
    
    def get_reaches_by_distance(
        self, 
        start_reach_id: int, 
        start_dist: float, 
        end_dist: float
    ) -> List[SwordReach]:
        """
        Get reaches within a distance range from outlet.
        
        Args:
            start_reach_id: Starting reach ID
            start_dist: Starting distance from outlet
            end_dist: Ending distance from outlet
        """
        sql = """
        WITH RECURSIVE reach_chain AS (
            SELECT r.*, 1 as depth
            FROM sword_reaches_v16 r
            WHERE reach_id = %(start_id)s
            
            UNION
            
            SELECT r.*, rc.depth + 1
            FROM sword_reaches_v16 r
            JOIN reach_chain rc ON r.reach_id = ANY(string_to_array(rc.rch_id_up, ' ')::bigint[])
            WHERE rc.depth < 100  -- Prevent infinite recursion
        )
        SELECT * FROM (
            SELECT DISTINCT r.*, n.dist_out as node_dist_out
            FROM reach_chain r
            JOIN sword_nodes_v16 n ON r.reach_id = n.reach_id
            WHERE n.dist_out BETWEEN %(end_dist)s AND %(start_dist)s
        ) sub
        ORDER BY sub.node_dist_out DESC;
        """
        
        reaches_df = gpd.read_postgis(
            sql,
            self.engine,
            params={
                'start_id': start_reach_id,
                'start_dist': start_dist,
                'end_dist': end_dist
            },
            geom_col='geometry'
        )
        
        return [self.get_reach(rid) for rid in reaches_df['reach_id'] if rid]
    
    def get_network_stats(self, reach_ids: List[int]) -> Dict[str, Any]:
        """
        Get statistics for a river network.
        
        Args:
            reach_ids: List of reach IDs in the network
        """
        sql = """
        SELECT 
            COUNT(DISTINCT r.reach_id) as reach_count,
            COUNT(DISTINCT n.node_id) as node_count,
            AVG(n.width) as mean_width,
            STDDEV(n.width) as std_width,
            AVG(r.slope) as mean_slope,
            STDDEV(r.slope) as std_slope,
            SUM(ST_Length(r.geometry::geography))/1000 as total_length_km
        FROM sword_reaches_v16 r
        JOIN sword_nodes_v16 n ON r.reach_id = n.reach_id
        WHERE r.reach_id = ANY(:reach_ids);
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(
                text(sql),
                {'reach_ids': reach_ids}
            ).first()
        
        return dict(result._mapping) if result else {}
    
    def batch_get_reaches(self, reach_ids: List[int]) -> List[SwordReach]:
        """
        Efficiently get multiple reaches in a single query.
        
        Args:
            reach_ids: List of reach IDs to fetch
        """
        if not reach_ids:
            return []
        
        # Get all reaches in one query
        reaches_sql = """
        SELECT * FROM sword_reaches_v16 
        WHERE reach_id = ANY(:reach_ids);
        """
        reaches_df = gpd.read_postgis(
            reaches_sql,
            self.engine,
            params={'reach_ids': reach_ids},
            geom_col='geometry'
        )
        
        # Get all nodes in one query
        nodes_sql = """
        SELECT * FROM sword_nodes_v16 
        WHERE reach_id = ANY(:reach_ids)
        ORDER BY reach_id, dist_out DESC;
        """
        nodes_df = gpd.read_postgis(
            nodes_sql,
            self.engine,
            params={'reach_ids': reach_ids},
            geom_col='geometry'
        )
        
        # Group nodes by reach_id
        nodes_by_reach = {rid: [] for rid in reach_ids}
        for _, row in nodes_df.iterrows():
            nodes_by_reach[row['reach_id']].append(SwordNode.from_gdf(row))
        
        # Create SwordReach objects
        reaches = []
        for _, row in reaches_df.iterrows():
            rid = row['reach_id']
            reaches.append(SwordReach(
                reach_id=rid,
                nodes=nodes_by_reach.get(rid, []),
                rch_id_up=[int(x) for x in str(row['rch_id_up']).split() if x],
                rch_id_dn=[int(x) for x in str(row['rch_id_dn']).split() if x],
                facc=float(row['facc']),
                geometry=row['geometry']
            ))
        
        return reaches