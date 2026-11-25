"""Arayüzün çekirdek fonksiyonları (Gradio olmadan)."""

import cv2
import numpy as np
import pytest

from floorplan_parser.akis import plani_coz
from floorplan_parser.arayuz import (
    ELLE,
    OTOMATIK,
    TARAMA,
    coz,
    foto_oku,
    graf_goruntusu,
    json_sozluk,
    onizleme,
    ozet,
    sonuc_goruntusu,
)
from floorplan_parser.foto_benzetim import FotoAyar, foto_uret
from floorplan_parser.sentetik import ev_ciz, plan_ciz


def _sayfa() -> np.ndarray:
    p = plan_ciz().goruntu
    sayfa = np.full((1400, 990), 255, np.uint8)
    y, x = (1400 - p.shape[0]) // 2, (990 - p.shape[1]) // 2
    sayfa[y : y + p.shape[0], x : x + p.shape[1]] = p
    return sayfa


@pytest.fixture(scope="module")
def foto():
    return foto_uret(_sayfa(), FotoAyar(egim=25, tohum=4))


def test_otomatik_kose_ve_ciktilar(foto):
    rgb = cv2.cvtColor(foto.goruntu, cv2.COLOR_GRAY2RGB)
    r = coz(rgb)
    assert r.kose_kaynagi == OTOMATIK
    assert r.sayfa.shape == (3508, 2480)
    assert len(r.plan.kapilar) >= 1
    sonuc, graf = sonuc_goruntusu(r), graf_goruntusu(r)
    assert sonuc.ndim == 3 and sonuc.shape[2] == 3 and max(sonuc.shape) <= 2000
    assert graf.ndim == 3 and graf.shape[2] == 3
    d = json_sozluk(r)
    assert d["input"]["mode"] == "auto" and len(d["input"]["paper_corners_in_photo"]) == 4
    assert "doors" in ozet(r)


def test_elle_isaretlenen_koseler_kullanilir(foto):
    karisik = [tuple(foto.kose[i]) for i in (2, 0, 3, 1)]
    r = coz(foto.goruntu, karisik)
    assert r.kose_kaynagi == ELLE
    assert np.abs(r.kose - foto.kose).max() < 1e-6


def test_tarama_kipi_duzeltme_yapmaz():
    sayfa = _sayfa()
    r = coz(sayfa, tarama=True)
    assert r.kose_kaynagi == TARAMA and r.kose is None
    assert r.sayfa.shape == sayfa.shape


def test_kagit_yoksa_hata():
    with pytest.raises(ValueError, match="corners"):
        coz(np.full((600, 800), 120, np.uint8))


def test_ornek_ev_planinin_hepsi_bulunur():
    e = ev_ciz()
    assert e.goruntu.shape == (2480, 3508)  # A4 yatay, 300 dpi
    s = plani_coz(e.goruntu)
    assert len(s.kapilar) == e.kapi_sayisi == 8
    assert len(s.pencereler) == e.pencere_sayisi == 13
    assert s.graf.number_of_nodes() - 1 == e.oda_sayisi == 8


def test_foto_oku_exif_yonunu_uygular(tmp_path):
    from PIL import Image

    yatay = np.zeros((60, 100, 3), np.uint8)
    yatay[:, :10] = 255  # sol kenar beyaz
    exif = Image.Exif()
    exif[0x0112] = 6  # 90 derece saat yönünde döndürülerek gösterilecek
    yol = tmp_path / "telefon.jpg"
    Image.fromarray(yatay).save(yol, exif=exif)
    g = foto_oku(yol)
    assert g.shape[:2] == (100, 60)
    assert g[:10].mean() > 200  # beyaz kenar artık üstte


def test_onizleme_olcegi_tiklamayi_geri_tasir(foto):
    g, k = onizleme(foto.goruntu, foto.kose, [(100.0, 200.0)])
    assert max(g.shape[:2]) <= 1600 and g.shape[2] == 3
    assert k == pytest.approx(1600 / max(foto.goruntu.shape))
