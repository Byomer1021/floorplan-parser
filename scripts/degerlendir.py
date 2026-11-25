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
from floorplan_parser.degerlendirme import plan_puani
from floorplan_parser.duvar import VARSAYILAN_INCE_KURAL
from floorplan_parser.oda import VARSAYILAN_ODA_KURAL
from floorplan_parser.onisleme import ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


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

        satirlar.append(
            {
                "ad": o.ad,
                "grup": o.grup
                if o.grup != "I-II"
                else ("I-II dolu" if dolgu >= 0.8 else "I-II bos"),
                "dolgu": round(dolgu, 3),
                **plan_puani(s, c, cevap_duvar),
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
