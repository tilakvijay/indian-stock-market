import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from indian_stock_pipeline.ui.streamlit_app import render


if __name__ == "__main__":
    render()
