"""Run with: python -m streamlit run app/app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dashboard import main

main()
