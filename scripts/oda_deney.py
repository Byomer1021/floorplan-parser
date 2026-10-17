"""CVC-FP'de oda grafı deneyi: odalar ve kapı kenarları, cevaptaki Room ve
``access`` ilişkileriyle.

Plan başına:
- oda_dogru / oda_fazla / oda_eksik: bulunan oda bölgesi ile cevaptaki Room
  poligonu IoU ≥ 0.5, bire bir.
- kenar_dogru / kenar_fazla / kenar_eksik: oda-oda kapı kenarları. Doğru:
  kenarın kapısı bir cevap kapısıyla eşleşmiş (poligon IoU ≥ 0.5) ve kenarın
  iki odası, o kapının ``access`` ilişkisindeki iki odayla eşleşmiş. Cevap
  kenarları: kapılı ``access`` ilişkileri (Door, Room, Room).
- ayrim: cevaptaki duvarsız (Separation) ``access`` sayısı; bu ayrımlar
  çizimde yok, iki oda tek bölge olur.
- dis_kapi: bir yanı bina dışı olan kapı kenarı sayısı (cevapta karşılığı yok,
  yalnız sayılır).

Yalnız I-II, dolu duvarlı planlar. Varsayılan bölüm geliştirme.
Kullanım: uv run python scripts/oda_deney.py [--etiket ad]
"""

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from floorplan_parser.aciklik import aciklik_bul, bosluklar, kose_bosluklari
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import bolge_eslesmeleri, parca_maskesi, poligon_eslesmeleri
from floorplan_parser.duvar import iki_olcekli_duvarlar
from floorplan_parser.oda import DIS, VARSAYILAN_ODA_KURAL, oda_grafi
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="gelistirme", choices=["gelistirme", "test"])
    ap.add_argument("--ayak-izi", type=int, default=VARSAYILAN_ODA_KURAL.ayak_izi_cekirdek)
    ap.add_argument(
        "--tum-bosluk",
        type=int,
        default=0,
        help="1: sınıflanmamış boşluklar da kapanır (geliştirmede odaları böldü, §9)",
    )
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()
    kural = replace(VARSAYILAN_ODA_KURAL, ayak_izi_cekirdek=a.ayak_izi)

    satirlar = []
    t_basla = time.perf_counter()
    for o in ornekler(bolum=a.bolum):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        ham = ikile(g)
        if (ham[maske(c, "Wall") > 0] > 0).mean() < 0.8:
            continue
        t0 = time.perf_counter()
        ikili = delik_doldur(ham, 4)
        kalin, kalin_p, ince_p = iki_olcekli_duvarlar(ikili)
        duvar = np.maximum(kalin, parca_maskesi(ince_p, g.shape))
        kapilar, pencereler, _ = aciklik_bul(ikili, kalin_p + ince_p, duvar)
        ek = bosluklar(kalin_p + ince_p, duvar) + kose_bosluklari(duvar) if a.tum_bosluk else []
        graf, etiket = oda_grafi(duvar, kapilar, pencereler, kural, ek)
        sure = time.perf_counter() - t0

        odalar = c.sinif("Room")
        kapilar_c = c.sinif("Door")
        oda_es = {
            e: odalar[j].id for e, j, _ in bolge_eslesmeleri(etiket, [n.noktalar for n in odalar])
        }
        kapi_es = {
            i: kapilar_c[j].id
            for i, j in poligon_eslesmeleri(
                [k.poligon() for k in kapilar], [n.noktalar for n in kapilar_c]
            )
        }
        sinif = {n.id: n.sinif for n in c.nesneler}
        cevap_kenar = {}  # kapı id → oda id çifti
        ayrim = 0
        for tip, ids in c.iliskiler:
            if tip != "access":
                continue
            turler = [sinif.get(i) for i in ids]
            oda_ids = frozenset(i for i in ids if sinif.get(i) == "Room")
            if "Door" in turler and len(oda_ids) == 2:
                cevap_kenar[next(i for i in ids if sinif.get(i) == "Door")] = oda_ids
            elif "Separation" in turler:
                ayrim += 1

        oda_oda = [(u, v, d["kapi"]) for u, v, d in graf.edges(data=True) if DIS not in (u, v)]
        dogru = sum(
            1
            for u, v, i in oda_oda
            if i in kapi_es
            and u in oda_es
            and v in oda_es
            and cevap_kenar.get(kapi_es[i]) == frozenset((oda_es[u], oda_es[v]))
        )
        n_oda = graf.number_of_nodes() - 1
        satirlar.append(
            {
                "ad": o.ad,
                "oda_dogru": len(oda_es),
                "oda_fazla": n_oda - len(oda_es),
                "oda_eksik": len(odalar) - len(oda_es),
                "kenar_dogru": dogru,
                "kenar_fazla": len(oda_oda) - dogru,
                "kenar_eksik": len(cevap_kenar) - dogru,
                "ayrim": ayrim,
                "dis_kapi": graf.number_of_edges() - len(oda_oda),
                "sure_s": round(sure, 2),
            }
        )
        s = satirlar[-1]
        print(
            f"{o.ad:24} oda {s['oda_dogru']}/{s['oda_fazla']}/{s['oda_eksik']}  "
            f"kenar {s['kenar_dogru']}/{s['kenar_fazla']}/{s['kenar_eksik']}  ayrım {ayrim}",
            flush=True,
        )

    ad = f"oda_deney_{a.bolum}{'_' + a.etiket if a.etiket else ''}.csv"
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    top = {k: sum(s[k] for s in satirlar) for k in satirlar[0] if k not in ("ad", "sure_s")}
    for tur in ("oda", "kenar"):
        d, fz, e = top[f"{tur}_dogru"], top[f"{tur}_fazla"], top[f"{tur}_eksik"]
        kes, duy = d / max(d + fz, 1), d / max(d + e, 1)
        print(
            f"{tur}: dogru {d} fazla {fz} eksik {e}  kesinlik {kes:.3f} duyarlilik {duy:.3f} "
            f"F1 {2 * kes * duy / max(kes + duy, 1e-9):.3f}"
        )
    print(f"ayrim (duvarsiz access) {top['ayrim']}, dis kapi kenari {top['dis_kapi']}")
    print(f"{len(satirlar)} plan, {time.perf_counter() - t_basla:.0f} s -> results/{ad}")


if __name__ == "__main__":
    main()
