import math

import cv2
import numpy as np
import pytest

from floorplan_parser.aciklik import (
    AciklikKural,
    aciklik_bul,
    birlesim_poligonu,
    bosluklar,
    genis_bosluklar,
)
from floorplan_parser.degerlendirme import parca_maskesi, poligon_eslestir, poligon_iou
from floorplan_parser.duvar import Parca, iki_olcekli_duvarlar
from floorplan_parser.onisleme import delik_doldur, ikile
from floorplan_parser.sentetik import plan_ciz


@pytest.mark.parametrize(
    "plan, k_kalin",
    [(plan_ciz(), 5), (plan_ciz(duvar=24, ic_duvar=8, merdiven=True), 13)],
)
def test_sentetik_kapi_ve_pencere(plan, k_kalin):
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=k_kalin)
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    # Sentetik kapı 60 px: CVC-FP ölçeğindeki 75 px kapı sınırı burada 40.
    kural = AciklikKural(kapi_min_yaricap=40)
    kapilar, pencereler, kalan = aciklik_bul(ikili, kalin_p + ince_p, duvar, kural)
    assert len(kapilar) == 1 and len(pencereler) == 1 and kalan == []
    k = kapilar[0]
    assert k.skor == 1.0
    assert abs(k.yaricap - plan.kapilar[0].genislik) <= 6
    # Yay menteşeden sağa açılıyor (çizimde kanat menteşeden +x yönünde).
    assert k.acik_uc[0] - k.mentese[0] > 40
    p = pencereler[0]
    assert p.cizgi == 3
    assert abs(p.bosluk.genislik - 100) <= 2


def test_kucuk_yay_kanatli_pencere_sayilir():
    # Varsayılan sınır 75 px: 60 px'lik yay kanatlı pencere. Poligonu açılma
    # alanı + boşluk dikdörtgeni (kanat_dikdortgen), bayrak kapalıyken açılma alanı.
    plan = plan_ciz()
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=5)
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    kapilar, pencereler, _ = aciklik_bul(ikili, kalin_p + ince_p, duvar)
    assert kapilar == []
    kanatli = [p for p in pencereler if p.kanat is not None]
    assert len(kanatli) == 1 and len(pencereler) == 2
    k = kanatli[0]
    alan = cv2.contourArea(k.poligon().astype(np.float32))
    ceyrek = cv2.contourArea(k.kanat.poligon().astype(np.float32))
    dikdortgen = k.bosluk.genislik * k.bosluk.kalinlik
    assert alan == pytest.approx(ceyrek + dikdortgen, rel=0.1)
    _, pencereler, _ = aciklik_bul(
        ikili, kalin_p + ince_p, duvar, AciklikKural(kanat_dikdortgen=False)
    )
    k = next(p for p in pencereler if p.kanat is not None)
    assert np.array_equal(k.poligon(), k.kanat.poligon())


def test_ince_yaricap_taramasi_kayik_yayi_bulur():
    # Boşluk 120 px, yay yarıçapı 97 (0.81 ×): kaba tarama (0.85 × en küçük)
    # menteşe kaydırmasıyla bile tutturamaz; kapı boyundaki ek ölçekler
    # (0.83 ×) yakalar. Sentetik ince duvarda kaba taramanın kaçırdığı tek
    # aralık bu alt uç; gerçek planda kalınlıksız köşe adaylarında aralar da
    # kaçıyor (olcumler §15).
    plan = plan_ciz(kapi_genislik=120)
    mx, my = plan.kapilar[0].mentese
    cv2.ellipse(plan.goruntu, (mx, my), (120, 120), 0, 0, 90, 255, 4)
    cv2.ellipse(plan.goruntu, (mx, my), (97, 97), 0, 0, 90, 0, 2)
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=5)
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    parcalar = kalin_p + ince_p
    kaba, _, _ = aciklik_bul(ikili, parcalar, duvar, AciklikKural(kapi_ek_olcekleri=()))
    assert kaba == []
    ince, _, _ = aciklik_bul(ikili, parcalar, duvar)
    assert len(ince) == 1 and ince[0].yaricap == pytest.approx(97, abs=6)


def test_cift_kanatli_pencere():
    # Sentetik pencereye (100-200, y 350, duvar 10) iki kanat: menteşeler iç
    # yüzde (y 345) iki uçta, yarıçap 50, odaya (yukarı) açılıp ortada buluşur.
    plan = plan_ciz()
    for merkez, aci in (((100, 345), 270), ((200, 345), 180)):
        cv2.ellipse(plan.goruntu, merkez, (50, 50), 0, aci, aci + 90, 0, 2)
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=5)
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    kapilar, pencereler, _ = aciklik_bul(
        ikili, kalin_p + ince_p, duvar, AciklikKural(kapi_min_yaricap=40, cift_kanat=True)
    )
    assert len(kapilar) == 1 and len(pencereler) == 1
    p = pencereler[0]
    assert len(p.kanatlar) == 2 and p.kanatlar[0].skor >= 0.8
    # Poligon: 100x10 dikdörtgen + iki çeyrek daire (yukarıda).
    beklenen = 100 * 10 + 2 * math.pi / 4 * 50**2
    assert cv2.contourArea(p.poligon().astype(np.float32)) == pytest.approx(beklenen, rel=0.08)
    assert p.poligon()[:, 1].min() == pytest.approx(345 - 50, abs=6)
    # Bayrak kapalıyken aynı boşluk paralel çizgili pencere.
    _, pencereler, _ = aciklik_bul(
        ikili, kalin_p + ince_p, duvar, AciklikKural(kapi_min_yaricap=40, cift_kanat=False)
    )
    assert len(pencereler) == 1 and pencereler[0].kanatlar == () and pencereler[0].cizgi == 3


def test_birlesim_poligonu():
    kare = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    p = birlesim_poligonu([kare, kare + [5, 0]])
    assert cv2.contourArea(p.astype(np.float32)) == pytest.approx(150, rel=0.03)


def _dikdortgen_parca(x1, y1, x2, y2, kalinlik, kapsam=None):
    return Parca(x1, y1, x2, y2, kalinlik, kapsam or (x1, y1, x2, y2))


def test_kose_dibindeki_kapi_bulunur_t_kavsagi_bosluk_acmaz():
    # V1 yukarıdan gelip H'nin üstünde biter (uç kavşağa çekilmiş, kapsam H'nin
    # alt yüzüne kadar). Altında 56 px boşluk, sonra aynı hatta V2.
    v1 = _dikdortgen_parca(300, 50, 300, 180, 8.0, (300, 46, 300, 184))
    h = _dikdortgen_parca(300, 180, 500, 180, 8.0, (296, 180, 500, 180))
    v2 = _dikdortgen_parca(300, 240, 300, 350, 8.0)
    # T-kavşağı: V3, H'ye üstten dayanır; altta 182 px ötede dik H2 var.
    v3 = _dikdortgen_parca(450, 50, 450, 180, 8.0, (450, 46, 450, 184))
    h2 = _dikdortgen_parca(400, 370, 560, 370, 8.0)
    parcalar = [v1, h, v2, v3, h2]
    duvar = parca_maskesi(parcalar, (400, 600))
    bulunan = bosluklar(parcalar, duvar)
    assert len(bulunan) == 1
    b = bulunan[0]
    assert abs(b.a[0] - 300) < 1 and abs(b.b[0] - 300) < 1
    assert sorted([b.a[1], b.b[1]]) == pytest.approx([184.5, 239.5], abs=1.5)


def test_genis_bosluk_kisa_ayak_kabul_dik_duvar_ret():
    # Yatay duvar parçası 50-300 (kalınlık 20); 220 px ötede parçaya dönüşmemiş
    # kısa ayak (520-540, aynı bantta). Dik uzun duvara çarpan ışın boşluk değil.
    h = _dikdortgen_parca(50, 200, 300, 200, 20.0)
    ayak = parca_maskesi([h], (400, 700))
    ayak[190:210, 520:540] = 255
    assert genis_bosluklar([h], ayak) == []
    bulunan = genis_bosluklar([h], ayak, kisa_duvar=True)
    assert len(bulunan) == 1 and bulunan[0].genislik == pytest.approx(220, abs=3)
    dik = parca_maskesi([h], (400, 700))
    dik[50:350, 520:540] = 255
    assert genis_bosluklar([h], dik, kisa_duvar=True) == []


def test_yan_suzgec_kesik_hattaki_uctan_kirisi_atlar():
    # Koridor: üst duvar y=80; alt duvar y=200, bölmenin iki yanında kapı
    # boşluklarıyla kesik (240-360 arası). Bölme (x=300) alt duvar hattında
    # bitiyor; ucundan yukarı ışın koridoru boydan kesip üst duvara çarpar.
    ust = _dikdortgen_parca(50, 80, 550, 80, 10.0)
    alt_sol = _dikdortgen_parca(50, 200, 240, 200, 10.0)
    alt_sag = _dikdortgen_parca(360, 200, 550, 200, 10.0)
    bolme = _dikdortgen_parca(300, 200, 300, 380, 10.0)
    parcalar = [ust, alt_sol, alt_sag, bolme]
    duvar = parca_maskesi(parcalar, (400, 600))

    def koridoru_kesen(bs):
        return [b for b in bs if abs(b.a[0] - 300) < 2 and abs(b.b[0] - 300) < 2]

    assert len(koridoru_kesen(bosluklar(parcalar, duvar))) == 1
    suzulmus = bosluklar(parcalar, duvar, yan_suzgec="duvarli")
    assert koridoru_kesen(suzulmus) == []
    # Kapı boşlukları (alt duvar uçlarından bölmeye) kalır.
    assert sum(abs(b.a[1] - 200) < 2 for b in suzulmus) == 2


def test_poligon_iou_ve_esleme():
    kare = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    kayik = kare + [5, 0]
    assert poligon_iou(kare, kare) == pytest.approx(1.0, abs=0.02)
    assert poligon_iou(kare, kayik) == pytest.approx(1 / 3, abs=0.03)
    assert poligon_iou(kare, kare + [100, 0]) == 0.0
    assert poligon_eslestir([kare, kare + [100, 0]], [kayik + [0, 0], kare + [0, 1]], 0.5) == (
        1,
        1,
        1,
    )
