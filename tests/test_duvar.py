import cv2
import numpy as np
import pytest

from floorplan_parser.degerlendirme import iou, parca_eslestir, parca_maskesi
from floorplan_parser.duvar import Parca, duvarlari_bul
from floorplan_parser.onisleme import duvar_adayi, ikile, kalin_ayir, kapa
from floorplan_parser.sentetik import SentetikPlan, plan_ciz


def _cevap(p: SentetikPlan, kaydir=(0.0, 0.0)) -> list[tuple[float, ...]]:
    # Sentetik çizimde duvar [y-h, y+h) satırlarında: piksel merkezi koordinatında
    # orta çizgi y - 0.5.
    dx, dy = kaydir
    return [
        (d.x1 - 0.5 + dx, d.y1 - 0.5 + dy, d.x2 - 0.5 + dx, d.y2 - 0.5 + dy) for d in p.duvarlar
    ]


@pytest.mark.parametrize("kalinlik", [10, 16, 22])
def test_sentetik_duvarlarin_hepsi_bulunur(kalinlik):
    p = plan_ciz(duvar=kalinlik)
    parcalar = duvarlari_bul(kalin_ayir(ikile(p.goruntu), 5)[0])
    assert parca_eslestir(parcalar, _cevap(p), uc_tol=2) == (7, 0, 0)
    assert {x.kalinlik for x in parcalar} == {kalinlik}
    assert iou(parca_maskesi(parcalar, p.goruntu.shape), p.duvar_maskesi) > 0.999


@pytest.mark.parametrize("aci", [7, 15, 30, 45, 60])
def test_donuk_planda_duvarlar_bulunur(aci):
    p = plan_ciz()
    g = cv2.copyMakeBorder(p.goruntu, 250, 250, 150, 150, cv2.BORDER_CONSTANT, value=255)
    M = cv2.getRotationMatrix2D((450, 450), aci, 1.0)
    g = cv2.warpAffine(g, M, (900, 900), flags=cv2.INTER_NEAREST, borderValue=255)
    cevap = []
    for x1, y1, x2, y2 in _cevap(p, kaydir=(150, 250)):
        cevap.append((*(M @ [x1, y1, 1]), *(M @ [x2, y2, 1])))
    parcalar = duvarlari_bul(kalin_ayir(ikile(g), 5)[0])
    assert parca_eslestir(parcalar, cevap, uc_tol=2) == (7, 0, 0)
    assert all(abs(x.kalinlik - 10) <= 0.5 for x in parcalar)


def test_max_bosluk_kapidan_genisse_kapi_duvar_olur():
    p = plan_ciz()  # kapı 60 piksel
    kalin = kalin_ayir(ikile(p.goruntu), 5)[0]
    assert parca_eslestir(duvarlari_bul(kalin, max_bosluk=50), _cevap(p), 2) == (7, 0, 0)
    # İç duvarın iki parçası tek parça olur: 1 fazla, 2 eksik.
    assert parca_eslestir(duvarlari_bul(kalin, max_bosluk=70), _cevap(p), 2) == (5, 1, 2)


def test_kose_kesisim_noktasinda():
    p = plan_ciz()
    parcalar = duvarlari_bul(kalin_ayir(ikile(p.goruntu), 5)[0])
    uclar = np.array([[x.x1, x.y1] for x in parcalar] + [[x.x2, x.y2] for x in parcalar])
    # Sol üst köşe (orta çizgilerin kesişimi 49.5, 49.5) iki parçanın ucu.
    assert (np.linalg.norm(uclar - [49.5, 49.5], axis=1) < 1).sum() == 2


def test_gurultulu_planda_once_kapama_pencereyi_duvar_yapar():
    p = plan_ciz(gurultu=0.01, tohum=0)
    kalin = kalin_ayir(kapa(ikile(p.goruntu), 3), 5)[0]
    # Alt duvar pencereyle birleşip tek parça: 1 fazla, 2 eksik.
    assert parca_eslestir(duvarlari_bul(kalin), _cevap(p), 2) == (5, 1, 2)


@pytest.mark.parametrize("tohum", [0, 1, 2, 3])
def test_gurultulu_planda_varsayilan_akis(tohum):
    # Varsayılan akış (delik doldur → açma), 5 tohumun 4'ünde tam; tohum 4'te
    # 1 fazla parça (docs/olcumler.md §4).
    p = plan_ciz(gurultu=0.01, tohum=tohum)
    parcalar = duvarlari_bul(duvar_adayi(p.goruntu, 5))
    assert parca_eslestir(parcalar, _cevap(p), uc_tol=2) == (7, 0, 0)


def test_parca_maskesi_tek_parca():
    m = parca_maskesi([Parca(45, 49.5, 554, 49.5, 10.0, (45, 49.5, 554, 49.5))], (100, 600))
    ys, xs = np.nonzero(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (45, 54, 45, 554)
    assert len(ys) == 10 * 510


def test_parca_maskesi_ust_uste_binen_kose_dolu():
    a = Parca(10, 10, 60, 10, 6.0, (7, 10, 60, 10))
    b = Parca(10, 10, 10, 60, 6.0, (10, 7, 10, 60))
    m = parca_maskesi([a, b], (80, 80))
    assert m[8:13, 8:13].min() == 255
