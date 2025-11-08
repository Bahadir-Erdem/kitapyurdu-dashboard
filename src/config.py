from pathlib import Path
from typing import Final

# Root directory
ROOT_DIR: Final = Path(__file__).resolve().parents[1]
SRC_DIR: Final = ROOT_DIR / "src"
DATA_DIR: Final = ROOT_DIR / "data"

# Data files
BOOK_DATA_FILE_PATH: Final = DATA_DIR / "book_data_raw.json"
SAMPLED_BOOK_DATA_FILE_PATH: Final = DATA_DIR / "book_data_sample.parquet.gz"

# Configurationss
SAMPLE_SIZE: Final = 100000

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories() -> None:
    """Create all necessary directories."""
    directories = [DATA_DIR]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
