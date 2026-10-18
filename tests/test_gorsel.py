import json

import numpy as np

from floorplan_parser.akis import plani_coz
from floorplan_parser.gorsel import VARSAYILAN_LEJANT, graf_sozluk, katmanlar, lejant
from floorplan_parser.sentetik import plan_ciz


def test_katmanlar_ve_graf_json():
    # Sentetik plan (test_oda.test_plani_coz_sentetik ile aynı): 2 oda, 1 kapı.
    plan = plan_ciz(duvar=24, ic_duvar=8, kapi_genislik=90)
    s = plani_coz(plan.goruntu)
    g = katmanlar(plan.goruntu, s)
    assert g.shape == (*plan.goruntu.shape, 3) and g.dtype == np.uint8
    # Duvar pikselleri duvar rengine boyanmış: mavi kanal kırmızıdan büyük.
    duvar = s.duvar_maske > 0
    assert (g[duvar][:, 0] > g[duvar][:, 2]).mean() > 0.95
    lejant(g, VARSAYILAN_LEJANT)
    d = json.loads(json.dumps(graf_sozluk(s)))
    assert len(d["rooms"]) == 2 and len(d["doors"]) == 1
    assert d["edges"] == [{"door": 0, "rooms": [1, 2]}]
    assert d["windows"][0]["type"] == "parallel_lines"
