"""Configuration for ICESat-2 operations."""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sliderule
from sliderule import icesat2

@dataclass
class ICESat2Config:
    """Configuration for ICESat-2 processing."""
    endpoint: str = "slideruleearth.io"
    srt: int = icesat2.ATL08_WATER  # Water surface type
    cnf: int = 3  # High confidence
    len: int = 60  # Length of segment
    res: int = 30  # Resolution
    
    def to_params(self) -> Dict[str, Any]:
        """Convert config to sliderule parameters."""
        return {
            "srt": self.srt,
            "cnf": self.cnf,
            "len": self.len,
            "res": self.res
        } 