"""Database configuration."""
from dataclasses import dataclass
import os
from typing import Optional
from dotenv import load_dotenv

@dataclass
class DBConfig:
    """PostgreSQL database configuration."""
    dbname: str
    user: str
    password: str
    host: str
    port: str
    
    @classmethod
    def from_env(cls) -> 'DBConfig':
        """Create config from environment variables."""
        load_dotenv()
        
        required_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
            
        return cls(
            dbname=os.getenv('DB_NAME', ''),
            user=os.getenv('DB_USER', ''),
            password=os.getenv('DB_PASSWORD', ''),
            host=os.getenv('DB_HOST', ''),
            port=os.getenv('DB_PORT', '')
        )
    
    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}" 