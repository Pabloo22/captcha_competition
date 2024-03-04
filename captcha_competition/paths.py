"""Set the project path in the .env file."""
import os
import pathlib
from dotenv import load_dotenv


load_dotenv()

if "PROJECT_PATH" not in os.environ:
    raise ValueError("Please set the PROJECT_PATH in the .env file.")

PROJECT_PATH = pathlib.Path(os.getenv("PROJECT_PATH"))  # type: ignore
DATA_PATH = PROJECT_PATH / "data"
DATA_RAW_PATH = DATA_PATH / "raw"
DATA_PROCESSED_PATH = DATA_PATH / "processed"
MODELS_PATH = PROJECT_PATH / "models"
CONFIG_PATH = PROJECT_PATH / "config"
SCRIPTS_PATH = PROJECT_PATH / "scripts"
