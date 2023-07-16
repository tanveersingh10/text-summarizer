# using Python's dataclasses module to define a small data structure to hold configuration for data ingestion

from dataclasses import dataclass
from pathlib import Path

#decorator automatically adds methods like __init__ and __repr__
@dataclass(frozen=True) #frozen makes the data class immutable  
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
