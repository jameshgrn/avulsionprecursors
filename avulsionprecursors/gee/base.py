"""Base classes and utilities for Google Earth Engine operations."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import ee
import os
from dotenv import load_dotenv

@dataclass
class GEEConfig:
    """Configuration for Google Earth Engine."""
    service_account: str
    credentials_path: str
    bucket_name: str = 'leveefinders-test'
    
    @classmethod
    def from_env(cls) -> 'GEEConfig':
        """Create config from environment variables."""
        load_dotenv()
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        credentials_path = os.getenv('GEE_CREDENTIALS_PATH')
        
        if not service_account or not credentials_path:
            raise ValueError("Missing required GEE environment variables")
            
        return cls(
            service_account=service_account,
            credentials_path=credentials_path
        )

class GEEInitializer:
    """Handles Google Earth Engine initialization."""
    
    def __init__(self, config: GEEConfig):
        self.config = config
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize Google Earth Engine."""
        if self._initialized:
            return
            
        if not os.path.isfile(str(self.config.credentials_path)):
            raise FileNotFoundError(
                f"Credentials file not found at: {self.config.credentials_path}"
            )
            
        try:
            credentials = ee.ServiceAccountCredentials(
                self.config.service_account, 
                self.config.credentials_path
            )
            ee.Initialize(credentials)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GEE: {e}") 