"""Uçtan uca değerlendirme: ``akis.plani_coz`` + bütün ölçüler, tek geçiş.

Test bölümü bununla ve bir kez ölçülür (kararlar K-005). Geliştirmede ayrı
deney betiklerinin sayılarını vermesi gerekir (aynı ayarlar).

Plan başına: duvar maske ve parça IoU'su; kapı, pencere (poligon IoU ≥ 0.5)
ve oda (bölge IoU ≥ 0.5) doğru/fazla/eksik; kapı kenarları (oda_deney ile aynı
tanım); süre.

Yalnız I-II, dolu duvarlı planlar (cevap dolgusu ≥ 0.8; değerlendirme alt
kümesi, sistem stili bilmez).
Kullanım: uv run python scripts/degerlendir.py --bolum gelistirme
"""

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from floorplan_parser.aciklik import VARSAYILAN_ACIKLIK_KURAL
from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import (
    bolge_eslesmeleri,
    iou,
    parca_maskesi,
    poligon_eslesmeleri,
)
from floorplan_parser.duvar import VARSAYILAN_INCE_KURAL
from floorplan_parser.oda import DIS, VARSAYILAN_ODA_KURAL
from floorplan_parser.onisleme import ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def kutu(p: np.ndarray) -> np.ndarray:
    """Poligonun eksen hizalı sınır kutusu, 4 köşe (kutu IoU'lu eşleme için)."""
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def cevap_kenarlari(c) -> dict:
    sinif = {n.id: n.sinif for n in c.nesneler}
    kenar = {}
    for tip, ids in c.iliskiler:
        if tip != "access":
            continue
        oda_ids = frozenset(i for i in ids if sinif.get(i) == "Room")
        if any(sinif.get(i) == "Door" for i in ids) and len(oda_ids) == 2:
            kenar[next(i for i in ids if sinif.get(i) == "Door")] = oda_ids
    return kenar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", required=True, choices=["gelistirme", "test"])
    ap.add_argument("--kismi", default="", help="ESIK:KANAT_YAN:PENCEREDE, ör. 0.9:0.15:0")
    ap.add_argument("--etiket", default="")
    ap.add_argument(
        "--oda",
        default="",
        help="OdaKural alanları, ör. hizali_bosluk=1,ayak_izi_cekirdek=401",
    )
    ap.add_argument("--ince", default="", help="InceKural alanları, ör. yonsuz=1,min_uzunluk=30")
    ap.add_argument(
        "--stil",
        default="dolu",
        choices=["dolu", "hepsi"],
        help="dolu: I-II, cevap duvar dolgusu >= 0.8 (varsayılan); hepsi: bütün planlar",
    )
    a = ap.parse_args()
    aciklik_kural = VARSAYILAN_ACIKLIK_KURAL
    if a.kismi:
        e, y, pnc = a.kismi.split(":")
        aciklik_kural = replace(
            aciklik_kural,
            kismi_esik=float(e),
            kismi_kanat_yan=float(y),
            kismi_pencerede=bool(int(pnc)),
        )

    def kural_degistir(kural, metin: str):
        for parca in filter(None, metin.split(",")):
            ad, deger = parca.split("=")
            tip = type(getattr(kural, ad))
            kural = replace(kural, **{ad: tip(int(deger)) if tip is bool else tip(deger)})
        if metin:
            print("kural:", kural, flush=True)
        return kural

    oda_kural = kural_degistir(VARSAYILAN_ODA_KURAL, a.oda)
    ince_kural = kural_degistir(VARSAYILAN_INCE_KURAL, a.ince)

    satirlar = []
    t_basla = time.perf_counter()
    for o in ornekler(bolum=a.bolum):
        if a.stil == "dolu" and o.grup != "I-II":
            continue
        g, c = yukle(o)
        cevap_duvar = maske(c, "Wall")
        dolgu = float((ikile(g)[cevap_duvar > 0] > 0).mean())
        if a.stil == "dolu" and dolgu < 0.8:
            continue
        t0 = time.perf_counter()
        s = plani_coz(g, oda_kural=oda_kural, aciklik_kural=aciklik_kural, ince_kural=ince_kural)
        sure = time.perf_counter() - t0

        kapi_c, pencere_c, oda_c = c.sinif("Door"), c.sinif("Window"), c.sinif("Room")
        kapi_es = dict(
            poligon_eslesmeleri([k.poligon() for k in s.kapilar], [n.noktalar for n in kapi_c])
        )
        pencere_d = len(
            poligon_eslesmeleri(
                [p.poligon() for p in s.pencereler], [n.noktalar for n in pencere_c]
            )
        )
        kapi_kutu = len(
            poligon_eslesmeleri(
                [kutu(k.poligon()) for k in s.kapilar], [kutu(n.noktalar) for n in kapi_c]
            )
        )
        pencere_kutu = len(
            poligon_eslesmeleri(
                [kutu(p.poligon()) for p in s.pencereler], [kutu(n.noktalar) for n in pencere_c]
            )
        )
        oda_es = {
            e: oda_c[j].id
            for e, j, _ in bolge_eslesmeleri(s.oda_etiket, [n.noktalar for n in oda_c])
        }
        cevap_kenar = cevap_kenarlari(c)
        oda_oda = [(u, v, d["kapi"]) for u, v, d in s.graf.edges(data=True) if DIS not in (u, v)]
        kenar_d = sum(
            1
            for u, v, i in oda_oda
            if i in kapi_es
            and u in oda_es
            and v in oda_es
            and cevap_kenar.get(kapi_c[kapi_es[i]].id) == frozenset((oda_es[u], oda_es[v]))
        )
        n_oda = s.graf.number_of_nodes() - 1
        satirlar.append(
            {
                "ad": o.ad,
                "grup": o.grup
                if o.grup != "I-II"
                else ("I-II dolu" if dolgu >= 0.8 else "I-II bos"),
                "dolgu": round(dolgu, 3),
                "duvar_maske_iou": round(iou(s.duvar_maske, cevap_duvar), 4),
                "duvar_kesisim": int(((s.duvar_maske > 0) & (cevap_duvar > 0)).sum()),
                "duvar_birlesim": int(((s.duvar_maske > 0) | (cevap_duvar > 0)).sum()),
                "duvar_parca_iou": round(
                    iou(parca_maskesi(s.kalin_parcalar + s.ince_parcalar, g.shape), cevap_duvar), 4
                ),
                "kapi_dogru": len(kapi_es),
                "kapi_fazla": len(s.kapilar) - len(kapi_es),
                "kapi_eksik": len(kapi_c) - len(kapi_es),
                "kapi_kutu_dogru": kapi_kutu,
                "pencere_dogru": pencere_d,
                "pencere_fazla": len(s.pencereler) - pencere_d,
                "pencere_eksik": len(pencere_c) - pencere_d,
                "pencere_kutu_dogru": pencere_kutu,
                "oda_dogru": len(oda_es),
                "oda_fazla": n_oda - len(oda_es),
                "oda_eksik": len(oda_c) - len(oda_es),
                "kenar_dogru": kenar_d,
                "kenar_fazla": len(oda_oda) - kenar_d,
                "kenar_eksik": len(cevap_kenar) - kenar_d,
                "sure_s": round(sure, 2),
            }
        )
        print(o.ad, flush=True)

    ad = f"degerlendir_{a.bolum}{'_' + a.etiket if a.etiket else ''}.csv"
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)
    print(
        f"\n{a.bolum}: {len(satirlar)} plan, {time.perf_counter() - t_basla:.0f} s -> results/{ad}"
    )
    for alan in ("duvar_maske_iou", "duvar_parca_iou"):
        v = np.array([s[alan] for s in satirlar])
        print(
            f"{alan}: medyan {np.median(v):.3f} ortalama {v.mean():.3f} "
            f"[%10 {np.percentile(v, 10):.3f}, %90 {np.percentile(v, 90):.3f}]"
        )
    kes = sum(s["duvar_kesisim"] for s in satirlar)
    bir = sum(s["duvar_birlesim"] for s in satirlar)
    print(f"duvar IoU, bütün pikseller havuzlanmış: {kes / max(bir, 1):.3f}")
    for gr in sorted({s["grup"] for s in satirlar}):
        v = np.array([s["duvar_maske_iou"] for s in satirlar if s["grup"] == gr])
        print(f"  {gr:10} n={len(v):2} duvar IoU ortalama {v.mean():.3f}")
    for tur in ("kapi", "pencere", "oda", "kenar"):
        d = sum(s[f"{tur}_dogru"] for s in satirlar)
        fz = sum(s[f"{tur}_fazla"] for s in satirlar)
        e = sum(s[f"{tur}_eksik"] for s in satirlar)
        kes, duy = d / max(d + fz, 1), d / max(d + e, 1)
        print(
            f"{tur}: {d}/{fz}/{e}  K {kes:.3f} D {duy:.3f} F1 {2 * kes * duy / max(kes + duy, 1e-9):.3f}"
        )
        if tur in ("kapi", "pencere"):
            dk = sum(s[f"{tur}_kutu_dogru"] for s in satirlar)
            n_t, n_c = d + fz, d + e
            kk, dd = dk / max(n_t, 1), dk / max(n_c, 1)
            print(
                f"  {tur} (eksen hizalı kutu IoU >= 0.5): {dk}/{n_t - dk}/{n_c - dk}  "
                f"K {kk:.3f} D {dd:.3f} F1 {2 * kk * dd / max(kk + dd, 1e-9):.3f}"
            )
    print(f"sure medyan {np.median([s['sure_s'] for s in satirlar]):.1f} s")


if __name__ == "__main__":
    main()
