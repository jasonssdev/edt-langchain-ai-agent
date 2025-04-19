from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_DIR = DATA_DIR / "db"
DOCS_DIR = PROJECT_ROOT / "docs"
IMAGES_DIR = DOCS_DIR / "images"

# Specific files root
CHROMA_DB_PATH = DB_DIR / "chroma.db"
CONSULTORIO_PDF_PATH = RAW_DIR / "consultorio.pdf"
WORKFLOW_JPG_PATH = IMAGES_DIR / "workflow.jpg"