import numpy as np
import pytest

from floorplan_parser.cvcfp import (
    VARSAYILAN_KOK,
    ayrilan_test_adlari,
    cevap_oku,
    grup,
    maske,
    ornekler,
)

# Gerçek dosyalardaki biçim: class listesi, polygon, relation; varsayılan isim alanı.
SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" version="1.1">
<width>40</width>
<height>30</height>
<class>Wall</class>
<class>Door</class>
<polygon class="Wall" fill="#AFD8F8" id="0" transcription="" points="2,2 32,2 32,6 2,6 " />
<polygon class="Wall" fill="#AFD8F8" id="1" transcription="" points="2.5,2 6.5,2 6.5,22 2.5,22 " />
<polygon class="Door" fill="#F6BD0F" id="2" transcription="" points="10,6 14,6 14,10 10,10 " />
<relation type="incident" objects="0,1"/>
</svg>
"""


def test_cevap_oku(tmp_path):
    yol = tmp_path / "x_gt_3.svg"
    yol.write_text(SVG, encoding="utf-8")
    c = cevap_oku(yol)
    assert (c.genislik, c.yukseklik) == (40, 30)
    assert [n.sinif for n in c.nesneler] == ["Wall", "Wall", "Door"]
    assert c.nesneler[1].noktalar[0].tolist() == [2.5, 2.0]
    assert c.iliskiler == [("incident", (0, 1))]
    m = maske(c, "Wall")
    assert m.shape == (30, 40)
    assert m[4, 20] == 255 and m[15, 4] == 255 and m[15, 20] == 0
    assert m[4, 4] == 255  # iki duvarın üst üste bindiği köşe delik olmamalı
    assert maske(c, "Window").sum() == 0


def test_grup():
    assert grup("Ia_AP2022_sommaire") == "I-II"
    assert grup("IId_AD0801") == "I-II"
    assert grup("10") == "sayi"
    assert grup("image012") == "image"
    assert grup("p3") == "p"


@pytest.mark.skipif(not VARSAYILAN_KOK.is_dir(), reason="CVC-FP indirilmemiş")
def test_veri_seti_tam():
    o = ornekler()
    assert len(o) == 122
    sayac = {g: sum(x.grup == g for x in o) for g in ("I-II", "sayi", "image", "p")}
    assert sayac == {"I-II": 90, "sayi": 10, "image": 18, "p": 4}
    c = cevap_oku(o[0].cevap)
    assert len(c.sinif("Wall")) > 0 and np.isfinite(c.nesneler[0].noktalar).all()


@pytest.mark.skipif(not VARSAYILAN_KOK.is_dir(), reason="CVC-FP indirilmemiş")
def test_bolme_sabit_ve_ayrik():
    gel = {o.ad for o in ornekler(bolum="gelistirme")}
    test = {o.ad for o in ornekler(bolum="test")}
    assert not gel & test and len(gel | test) == 122
    assert len(test) == 36  # 27 + 3 + 5 + 1 (grup başına %30, yuvarlanmış)
    # Belirlenimci: yeniden hesaplanınca aynı küme.
    assert ayrilan_test_adlari(sorted(test | gel)) == test


def test_bolme_grup_basina():
    adlar = [f"image{i:03d}" for i in range(10)] + ["1", "2", "3"]
    t = ayrilan_test_adlari(adlar)
    assert sum(a.startswith("image") for a in t) == 3 and sum(a.isdigit() for a in t) == 1
