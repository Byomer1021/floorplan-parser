import numpy as np
import pytest

from floorplan_parser.aciklik import AciklikKural, aciklik_bul, bosluklar, kose_bosluklari
from floorplan_parser.degerlendirme import bolge_eslesmeleri, parca_maskesi
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import DIS, OdaKural, ayak_izi, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile
from floorplan_parser.sentetik import plan_ciz


def _akis(plan, k_kalin):
    ikili = delik_doldur(ikile(plan.goruntu), 4)
    kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili, k_kalin=k_kalin)
    parcalar = kalin_p + ince_p
    duvar = np.maximum(kalin, parca_maskesi(ince_p, ikili.shape))
    kapilar, pencereler, _ = aciklik_bul(ikili, parcalar, duvar, AciklikKural(kapi_min_yaricap=40))
    ek = bosluklar(parcalar, duvar) + kose_bosluklari(duvar)
    return duvar, kapilar, pencereler, ek


@pytest.mark.parametrize(
    "plan, k_kalin",
    [(plan_ciz(), 5), (plan_ciz(duvar=24, ic_duvar=8, merdiven=True), 13)],
)
def test_sentetik_iki_oda_bir_kapi(plan, k_kalin):
    duvar, kapilar, pencereler, ek = _akis(plan, k_kalin)
    g, etiket = oda_grafi(duvar, kapilar, pencereler, bosluklar=ek)
    assert sorted(n for n in g.nodes if n != DIS) == [1, 2]
    assert [(u, v) for u, v in g.edges()] == [(1, 2)]
    # İki oda: iç duvarın solu ve sağı; resmin kenarındaki şerit dış.
    assert (etiket[200, 150] > 0) and (etiket[200, 450] > 0)
    assert etiket[200, 150] != etiket[200, 450]
    assert etiket[5, 5] == DIS


def test_ince_duvar_kalin_duvara_yarik_birakmaz():
    # İnce iç duvar kalın dış duvardan 1 px genişletilmiş maskeyle kesiliyor;
    # kapsam çekilen uca uzamasaydı odalar o yarıktan birbirine akardı.
    duvar, kapilar, pencereler, ek = _akis(plan_ciz(duvar=24, ic_duvar=8), 13)
    g, _ = oda_grafi(duvar, kapilar, pencereler, bosluklar=ek)
    assert g.number_of_nodes() - 1 == 2


def test_ayak_izi_kapanmamis_aciklikta_ici_disari_akitmaz():
    # Kare oda, bir duvarında 100 px açıklık; açıklık kapatılmıyor.
    m = np.zeros((400, 400), np.uint8)
    m[50:60, 50:350] = m[340:350, 50:350] = m[50:350, 50:60] = 255
    m[50:150, 340:350] = m[250:350, 340:350] = 255
    eski, _ = oda_grafi(m, [], [], OdaKural(ayak_izi_cekirdek=0))
    yeni, etiket = oda_grafi(m, [], [], OdaKural(ayak_izi_cekirdek=151))
    assert eski.number_of_nodes() - 1 == 0  # iç mekân dışarı aktı
    assert yeni.number_of_nodes() - 1 == 1
    assert etiket[200, 200] == 1 and etiket[5, 5] == DIS
    a = ayak_izi(m, 151)
    assert a[200, 200] == 255 and a[5, 5] == 0


def test_bolge_eslesmeleri():
    etiket = np.zeros((100, 100), np.int32)
    etiket[10:50, 10:50] = 1
    etiket[60:90, 60:90] = 2
    kare = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], float)
    uzak = np.array([[0, 60], [20, 60], [20, 80], [0, 80]], float)
    ciftler = bolge_eslesmeleri(etiket, [kare, uzak])
    assert [(e, j) for e, j, _ in ciftler] == [(1, 0)]
    assert ciftler[0][2] > 0.9


def test_genis_bosluk_yalniz_ayni_dogrultuda():
    from floorplan_parser.aciklik import genis_bosluklar
    from floorplan_parser.duvar import Parca

    def p(x1, y1, x2, y2):
        return Parca(x1, y1, x2, y2, 10.0, (x1, y1, x2, y2))

    # Aynı doğruda 300 px açıklık (garaj kapısı gibi) ve karşı duvara dik
    # 300 px boşluk (odanın karşısı): yalnız ilki.
    sol, sag = p(20, 100, 200, 100), p(500, 100, 780, 100)
    dik_uc, karsi = p(100, 300, 100, 380), p(20, 690, 780, 690)
    parcalar = [sol, sag, dik_uc, karsi]
    duvar = parca_maskesi(parcalar, (800, 800))
    g = genis_bosluklar(parcalar, duvar)
    assert len(g) == 1
    assert abs(g[0].genislik - 300) <= 3


def test_plani_coz_sentetik():
    from floorplan_parser.akis import plani_coz

    plan = plan_ciz(duvar=24, ic_duvar=8, kapi_genislik=90)
    s = plani_coz(plan.goruntu)
    assert len(s.kapilar) == 1 and len(s.pencereler) == 1
    assert sorted(n for n in s.graf.nodes if n != DIS) == [1, 2]
    assert [(u, v) for u, v in s.graf.edges()] == [(1, 2)]
