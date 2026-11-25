import cv2
import numpy as np
import pytest

from floorplan_parser.foto import (
    duzelt,
    foto_ikile,
    homografi_uygula,
    isik_duzelt,
    kagit_bul,
    kose_hatasi,
    kose_sirala,
)
from floorplan_parser.foto_benzetim import FotoAyar, foto_uret
from floorplan_parser.sentetik import plan_ciz


def _sayfa() -> np.ndarray:
    """Sentetik plan, A4 oranlı beyaz sayfanın ortasında."""
    p = plan_ciz().goruntu
    sayfa = np.full((1400, 990), 255, np.uint8)
    y, x = (1400 - p.shape[0]) // 2, (990 - p.shape[1]) // 2
    sayfa[y : y + p.shape[0], x : x + p.shape[1]] = p
    return sayfa


def test_kose_sirala_donmus_dortgende():
    kose = np.array([[90, 10], [10, 0], [0, 80], [100, 95]], float)
    assert kose_sirala(kose).tolist() == [[10, 0], [90, 10], [100, 95], [0, 80]]


@pytest.mark.parametrize("egim", [0, 30])
@pytest.mark.parametrize("tohum", range(3))
def test_kagit_bul_koseleri_bir_iki_pikselde(egim, tohum):
    f = foto_uret(_sayfa(), FotoAyar(egim=egim, tohum=tohum))
    kose = kagit_bul(f.goruntu)
    assert kose is not None
    assert kose_hatasi(kose, f.kose) < 3


def test_duzelt_gercek_koselerle_sayfayi_geri_getirir():
    sayfa = _sayfa()
    f = foto_uret(sayfa, FotoAyar(egim=35, tohum=1))
    _, hom = duzelt(f.goruntu, f.kose, boyut=(990, 1400))
    p = np.array([[100, 100], [500, 700], [900, 1300]], float)
    assert np.abs(homografi_uygula(hom, f.tasi(p)) - p).max() < 0.5


def test_isik_duzelt_kagidi_esit_parlak_yapar():
    sayfa = _sayfa().astype(np.float32)
    x = np.linspace(1.0, 0.5, sayfa.shape[1], dtype=np.float32)
    golgeli = (sayfa * 0.9 * x[None]).astype(np.uint8)
    duz = isik_duzelt(golgeli)
    kagit = sayfa == 255
    assert golgeli[kagit].std() > 30
    assert duz[kagit].std() < 5
    assert duz[sayfa == 0].mean() < 60  # mürekkep koyu kalır


def test_foto_ikile_ince_soluk_cizgiyi_tutar():
    sayfa = np.full((300, 300), 250, np.uint8)
    sayfa[100:130, 20:280] = 40  # kalın duvar
    sayfa[200, 20:280] = 170  # bulanıklaşmış ince çizgi
    _, otsu = cv2.threshold(sayfa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    duz, siki = foto_ikile(sayfa)
    assert (otsu[200] > 0).sum() == 0  # Otsu ince çizgiyi kaybeder
    assert (duz[200, 20:280] == 0).all()  # kâğıda göre eşik tutar
    assert (siki[100:130, 20:280] > 0).all() and (siki[200] == 0).all()
