from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from a .env file, if it exists
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError as exc:
    logger.error(f"Failed to import or load dotenv: {exc}")

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# global constants
GPCR_CLASSES: dict[str, str] = {
    "A": "Class A (Rhodopsin)",
    "B1": "Class B1 (Secretin)",
    "B2": "Class B2 (Adhesion)",
    "C": "Class C (Glutamate)",
    "D1": "Class D1 (Ste2-like fungal pheromone)",
    "F": "Class F (Frizzled)",
    "O1": "Class O1 (fish-like odorant)",
    "O2": "Class O2 (tetrapod specific odorant)",
    "T2": "Class T2 (Taste 2)",
}
# alias table for class input normalization
ALIASES: dict[str, str] = {
    **{k: k for k in GPCR_CLASSES.keys()},  # "A" -> "A"
    **{k.lower(): k for k in GPCR_CLASSES.keys()},  # "a" -> "A"
    **{v.lower(): k for k, v in GPCR_CLASSES.items()},  # "class a (rhodopsin)" -> "A"
}

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
