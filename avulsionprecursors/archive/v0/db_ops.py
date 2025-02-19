# Database Operations
import geopandas as gpd
from sqlalchemy import create_engine

def create_db_engine(db_params):
    """
    Create a SQLAlchemy engine for connecting to a PostgreSQL database.

    Parameters:
    db_params (dict): A dictionary containing the database connection parameters.
                      Keys should include 'dbname', 'user', 'password', 'host', and 'port'.

    Returns:
    sqlalchemy.engine.base.Engine: A SQLAlchemy engine object for the PostgreSQL database.
    """
    return create_engine(f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

def get_dataframe(engine, sql):
    """
    Execute a SQL query and return the result as a GeoDataFrame.

    Parameters:
    engine (sqlalchemy.engine.base.Engine): A SQLAlchemy engine object for the PostgreSQL database.
    sql (str): The SQL query to execute.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the result of the SQL query.
    """
    return gpd.read_postgis(sql, engine, geom_col='geometry', crs='EPSG:4326')
