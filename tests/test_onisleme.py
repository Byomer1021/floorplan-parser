import cv2
import numpy as np
import pytest

from floorplan_parser.onisleme import ac, delik_doldur, ikile, kalin_ayir, kapa
from floorplan_parser.sentetik import MUREKKEP, plan_ciz


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return (a & b).sum() / (a | b).sum()


def test_ikile_murekkebi_on_plana_alir():
    p = plan_ciz()
    ikili = ikile(p.goruntu)
    assert np.array_equal(ikili > 0, p.goruntu == MUREKKEP)


@pytest.mark.parametrize("tohum", range(5))
def test_acma_tuz_gurultusunu_siler(tohum):
    temiz = ikile(plan_ciz().goruntu)
    gurultulu = ikile(plan_ciz(gurultu=0.01, tohum=tohum).goruntu)
    assert ((gurultulu > 0) & (temiz == 0)).sum() > 1000  # gürültü gerçekten var
    acik = ac(gurultulu, 3)
    # Çizimden uzaktaki gürültü tamamen gider. İnce çizgiye yapışan gürültü
    # 3x3'lük blok oluşturursa kalabilir (5 tohumda 0-2 piksel).
    yakin = cv2.dilate(temiz, np.ones((5, 5), np.uint8))
    assert ((acik > 0) & (yakin == 0)).sum() == 0


@pytest.mark.parametrize("tohum", range(5))
def test_once_kapama_duvar_deliklerini_doldurur(tohum):
    p = plan_ciz(gurultu=0.01, tohum=tohum)
    ikili = ikile(p.goruntu)
    duvar = p.duvar_maskesi > 0
    assert (ikili[duvar] == 0).sum() > 100  # duvarın içinde delik var
    # Önce açma: her delik çevresini de götürür, duvar eksik kalır.
    kalin, _ = kalin_ayir(ikili, 5)
    assert (kalin[duvar] == 0).sum() > 3 * (ikili[duvar] == 0).sum()
    # Önce kapama: duvarda eksik kalmaz.
    kalin, _ = kalin_ayir(kapa(ikili, 3), 5)
    assert (kalin[duvar] == 0).sum() == 0


def test_once_kapama_pencereyi_duvara_cevirir():
    # Pencerenin paralel çizgileri arası 1-2 piksel; 3x3 kapama bunları
    # birleştirir ve kalın/ince ayrımında pencere duvar olarak çıkar.
    p = plan_ciz()
    pc = p.pencereler[0]
    h = p.duvarlar[0].kalinlik // 2
    ikili = ikile(p.goruntu)

    def pencerede_kalin(m: np.ndarray) -> float:
        return (kalin_ayir(m, 5)[0][pc.y - h : pc.y + h, pc.x1 + 3 : pc.x2 - 3] > 0).mean()

    assert pencerede_kalin(ikili) == 0
    assert pencerede_kalin(kapa(ikili, 3)) == 1


def test_kalin_ayir_duvari_ince_cizgiden_ayirir():
    p = plan_ciz()
    kalin, ince = kalin_ayir(ikile(p.goruntu), 5)
    assert _iou(kalin, p.duvar_maskesi) > 0.99
    # İnce çizgiler (kapı yayı, kanat, pencere) neredeyse tamamen ince tarafta.
    ince_gercek = (p.goruntu == MUREKKEP) & (p.duvar_maskesi == 0)
    assert (ince[ince_gercek] > 0).mean() > 0.95
    # Kalın taraf duvardan en fazla çekirdek boyu kadar taşar.
    genis = cv2.dilate(p.duvar_maskesi, np.ones((5, 5), np.uint8))
    assert ((kalin > 0) & (genis == 0)).sum() == 0


def test_kapi_boslugu_kalin_maskede_acik_kalir():
    p = plan_ciz()
    kapi = p.kapilar[0]
    mx, my = kapi.mentese
    kalin, _ = kalin_ayir(ikile(p.goruntu), 5)
    bosluk = kalin[my + 3 : my + kapi.genislik - 3, mx - 5 : mx + 5]
    assert bosluk.sum() == 0


def test_genis_kapama_kapiyi_kapatir():
    # Kapama çekirdeği kapı genişliğini aşınca kapı duvar gibi görünür.
    p = plan_ciz()
    kapi = p.kapilar[0]
    mx, my = kapi.mentese
    kalin, _ = kalin_ayir(ikile(p.goruntu), 5)

    def bosluk_dolu(cekirdek: int) -> float:
        k = kapa(kalin, cekirdek)
        return (k[my + 3 : my + kapi.genislik - 3, mx - 5 : mx + 5] > 0).mean()

    assert bosluk_dolu(9) == 0
    assert bosluk_dolu(kapi.genislik + 5) == 1


def test_delik_doldur_kucuk_kapali_bolgeyi_doldurur():
    m = np.zeros((40, 40), np.uint8)
    m[5:35, 5:35] = 255
    m[10, 10] = 0  # 1 piksellik delik
    m[20:30, 20:30] = 0  # 100 piksellik kapalı bölge
    m[0:3, 0:2] = 0  # kenara değen kâğıt zaten 0
    d = delik_doldur(m, 4)
    assert d[10, 10] == 255
    assert d[20:30, 20:30].max() == 0
    assert d[0, 0] == 0


def test_delik_doldur_pencere_seritlerini_korur():
    # Temiz plandaki kapalı kâğıt bölgelerinin en küçüğü pencere şeritleri (100, 200).
    ikili = ikile(plan_ciz().goruntu)
    assert np.array_equal(delik_doldur(ikili, 99), ikili)
