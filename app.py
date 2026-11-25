"""Hugging Face Spaces girişi (kararlar K-012, K-013).

Space bu dosyayı çalıştırır. Yerelde ``scripts/arayuz.py`` kullanılır.
Paket kurulmadığı için ``src/`` yola ekleniyor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from floorplan_parser.arayuz import uygulama

uygulama().launch()
