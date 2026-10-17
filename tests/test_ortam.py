"""Ortam: çekirdek kütüphaneler yükleniyor, öğrenilmiş model kütüphanesi yok (K-002)."""

import re
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]

# Derin öğrenme çatıları ve hazır nesne tespit / segmentasyon kütüphaneleri.
YASAK = {
    "torch",
    "torchvision",
    "tensorflow",
    "tensorflow-cpu",
    "keras",
    "jax",
    "jaxlib",
    "ultralytics",
    "onnxruntime",
    "paddlepaddle",
    "detectron2",
    "mmdet",
    "timm",
    "transformers",
    "segment-anything",
}


def test_cekirdek_kutuphaneler_yukleniyor():
    import cv2
    import matplotlib
    import networkx
    import scipy
    import skimage

    assert int(cv2.__version__.split(".")[0]) >= 4
    for m in (matplotlib, networkx, scipy, skimage):
        assert m.__version__


def test_ogrenilmis_model_kutuphanesi_yok():
    kilit = (KOK / "uv.lock").read_text(encoding="utf-8")
    paketler = set(re.findall(r'^name = "([^"]+)"', kilit, re.MULTILINE))
    assert "opencv-python" in paketler  # kilit dosyası gerçekten okundu
    assert paketler.isdisjoint(YASAK), sorted(paketler & YASAK)
