import numpy as np
import pytest

from floorplan_parser.aciklik import AciklikKural, aciklik_bul, bosluklar
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
    # Varsayılan sınır 75 px: 60 px'lik yay kanatlı pencere, poligonu açılma alanı.
    plan = plan_ciz()
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=5)
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    kapilar, pencereler, _ = aciklik_bul(ikili, kalin_p + ince_p, duvar)
    assert kapilar == []
    kanatli = [p for p in pencereler if p.kanat is not None]
    assert len(kanatli) == 1 and len(pencereler) == 2
    assert np.array_equal(kanatli[0].poligon(), kanatli[0].kanat.poligon())


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
