"""CVC-FP veri seti: görüntü/cevap eşleme, SVG cevabını okuma, sınıf maskesi.

Arşiv düzeni: ``ImagesGT/<ad>.png|jpg`` ve ``ImagesGT/<ad>_gt_<n>.svg``.
123 görüntü, 122 cevap (``image001a``'nın cevabı yok).

Cevap SVG'si standart dışı: her nesne ``class`` özniteliği taşıyan bir
``<polygon>`` (Wall, Door, Window, Room, Separation, Parking), nesneler arası
ilişkiler ``<relation type=... objects="a,b">``. Poligon koordinatları
görüntü pikseli, ondalıklı.

Gruplar dosya adından (data/README.md): ``I``/``II`` önekli 90 plan, sayı
adlı 10, ``image`` önekli 18, ``p`` önekli 4. Çizim stilleri ölçümü
docs/olcumler.md §3.
"""

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.goruntu import oku

VARSAYILAN_KOK = Path(__file__).resolve().parents[2] / "data" / "raw" / "cvc-fp" / "ImagesGT"
_NS = "{http://www.w3.org/2000/svg}"
_ALT_PIKSEL = 4  # fillPoly için 2^4 = 1/16 piksel hassasiyet


@dataclass(frozen=True)
class Ornek:
    ad: str
    grup: str
    goruntu: Path
    cevap: Path


@dataclass(frozen=True)
class Nesne:
    id: int
    sinif: str
    noktalar: np.ndarray  # (n, 2), x y


@dataclass
class Cevap:
    genislik: int
    yukseklik: int
    nesneler: list[Nesne] = field(default_factory=list)
    iliskiler: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def sinif(self, ad: str) -> list[Nesne]:
        return [n for n in self.nesneler if n.sinif == ad]


def grup(ad: str) -> str:
    if ad.isdigit():
        return "sayi"
    if ad.startswith("image"):
        return "image"
    if re.fullmatch(r"p\d+", ad):
        return "p"
    if ad.startswith(("I", "II")):
        return "I-II"
    raise ValueError(f"bilinmeyen grup: {ad}")


def ornekler(kok: str | Path = VARSAYILAN_KOK, bolum: str | None = None) -> list[Ornek]:
    """Cevabı olan görüntüler, ada göre sıralı. ``bolum``: "gelistirme" ya da "test"."""
    kok = Path(kok)
    if not kok.is_dir():
        raise FileNotFoundError(f"CVC-FP bulunamadı: {kok} (data/README.md)")
    goruntuler = {p.stem: p for p in kok.iterdir() if p.suffix.lower() in (".png", ".jpg")}
    cevaplar = {re.sub(r"_gt_\d+$", "", p.stem): p for p in kok.glob("*.svg")}
    hepsi = [
        Ornek(ad, grup(ad), goruntuler[ad], cevaplar[ad])
        for ad in sorted(goruntuler.keys() & cevaplar.keys())
    ]
    if bolum is None:
        return hepsi
    testte = ayrilan_test_adlari([o.ad for o in hepsi])
    if bolum == "test":
        return [o for o in hepsi if o.ad in testte]
    if bolum == "gelistirme":
        return [o for o in hepsi if o.ad not in testte]
    raise ValueError(f"bilinmeyen bölüm: {bolum}")


def ayrilan_test_adlari(adlar: list[str], pay: float = 0.3) -> set[str]:
    """Sabit test bölümü (kararlar K-005): her gruptan ``pay`` kadarı, ada göre
    sha1 sırasıyla. Eşikler ve yöntem seçimi yalnız geliştirme bölümünde."""
    gruplar: dict[str, list[str]] = {}
    for ad in adlar:
        gruplar.setdefault(grup(ad), []).append(ad)
    secilen = set()
    for g in gruplar.values():
        g.sort(key=lambda a: hashlib.sha1(a.encode()).hexdigest())
        secilen.update(g[: round(len(g) * pay)])
    return secilen


def cevap_oku(yol: str | Path) -> Cevap:
    kok = ET.parse(yol).getroot()
    cevap = Cevap(int(kok.findtext(_NS + "width")), int(kok.findtext(_NS + "height")))
    for p in kok.iter(_NS + "polygon"):
        noktalar = np.array(
            [[float(v) for v in xy.split(",")] for xy in p.get("points").split()], np.float64
        )
        cevap.nesneler.append(Nesne(int(p.get("id")), p.get("class"), noktalar))
    for r in kok.iter(_NS + "relation"):
        nesneler = tuple(int(i) for i in r.get("objects").split(","))
        cevap.iliskiler.append((r.get("type"), nesneler))
    return cevap


def maske(cevap: Cevap, sinif: str) -> np.ndarray:
    """``sinif``'taki poligonların doldurulmuş maskesi (255), görüntü boyutunda."""
    m = np.zeros((cevap.yukseklik, cevap.genislik), np.uint8)
    for n in cevap.sinif(sinif):
        # Tek tek: fillPoly birden çok poligonu tek-çift kuralıyla doldurur ve
        # cevapta duvar poligonları köşelerde üst üste biniyor.
        p = np.round(n.noktalar * (1 << _ALT_PIKSEL)).astype(np.int32)
        cv2.fillPoly(m, [p], 255, cv2.LINE_8, _ALT_PIKSEL)
    return m


def yukle(ornek: Ornek) -> tuple[np.ndarray, Cevap]:
    """Gri görüntü ve cevap. Boyutlar tutmazsa hata."""
    g = oku(ornek.goruntu)
    c = cevap_oku(ornek.cevap)
    if g.shape != (c.yukseklik, c.genislik):
        raise ValueError(f"{ornek.ad}: görüntü {g.shape}, cevap {(c.yukseklik, c.genislik)}")
    return g, c
