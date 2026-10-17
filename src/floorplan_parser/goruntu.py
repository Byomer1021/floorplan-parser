"""Görüntü okuma ve yazma.

cv2.imread/imwrite Windows'ta ASCII olmayan yolları ("Masaüstü", "çizim.png")
açamıyor ve hata vermeden None döndürüyor. Bu yüzden dosya bayt olarak okunup
imdecode/imencode ile çözülüyor.
"""

from pathlib import Path

import cv2
import numpy as np


def oku(yol: str | Path, gri: bool = True) -> np.ndarray:
    yol = Path(yol)
    veri = np.fromfile(yol, dtype=np.uint8)
    goruntu = cv2.imdecode(veri, cv2.IMREAD_GRAYSCALE if gri else cv2.IMREAD_COLOR)
    if goruntu is None:
        raise ValueError(f"görüntü çözülemedi: {yol}")
    return goruntu


def yaz(yol: str | Path, goruntu: np.ndarray) -> None:
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    tamam, tampon = cv2.imencode(yol.suffix, goruntu)
    if not tamam:
        raise ValueError(f"görüntü kodlanamadı: {yol}")
    tampon.tofile(yol)
