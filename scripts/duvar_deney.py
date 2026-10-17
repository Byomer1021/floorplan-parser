"""CVC-FP'de duvar deneyi: kalın maske ve Hough parçaları, cevaptaki duvarla.

Ölçülen, plan başına:
- maske_iou: ``duvar_adayi`` (ikile → delik doldur → açma) ile cevabın IoU'su
- parca_iou: bulunan parçaların ``parca_maskesi``'nin cevapla IoU'su
- parca: bulunan parça sayısı, cevap_duvar: cevaptaki duvar poligonu sayısı
- sure_s: plan başına süre (okuma hariç)

Kullanım:
  uv run python scripts/duvar_deney.py --kalinlik 5 7 9 11 13 --yalniz-maske
  uv run python scripts/duvar_deney.py --kalinlik 9 --max-bosluk 20
Varsayılan bölüm geliştirme; test yalnız --bolum test ile, sonda bir kez.
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import iou, parca_maskesi
from floorplan_parser.duvar import duvarlari_bul
from floorplan_parser.onisleme import duvar_adayi

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="gelistirme", choices=["gelistirme", "test"])
    ap.add_argument("--grup", nargs="*", default=None, help="ör. I-II sayi image p")
    ap.add_argument("--kalinlik", type=int, nargs="+", default=[9])
    ap.add_argument("--max-delik", type=int, default=4)
    ap.add_argument("--max-bosluk", type=int, default=20)
    ap.add_argument("--min-uzunluk", type=int, default=15)
    ap.add_argument("--yalniz-maske", action="store_true")
    ap.add_argument("--etiket", default="")
    a = ap.parse_args()

    secilen = [o for o in ornekler(bolum=a.bolum) if a.grup is None or o.grup in a.grup]
    satirlar = []
    t_basla = time.perf_counter()
    for o in secilen:
        g, c = yukle(o)
        cevap = maske(c, "Wall")
        for k in a.kalinlik:
            t0 = time.perf_counter()
            kalin = duvar_adayi(g, k, a.max_delik)
            satir = {
                "ad": o.ad,
                "grup": o.grup,
                "kalinlik": k,
                "maske_iou": round(iou(kalin, cevap), 4),
                "cevap_duvar": len(c.sinif("Wall")),
            }
            if not a.yalniz_maske:
                parcalar = duvarlari_bul(kalin, a.max_bosluk, min_uzunluk=a.min_uzunluk)
                satir["parca"] = len(parcalar)
                satir["parca_iou"] = round(iou(parca_maskesi(parcalar, g.shape), cevap), 4)
            satir["sure_s"] = round(time.perf_counter() - t0, 2)
            satirlar.append(satir)
        print(
            f"{o.ad:24} "
            + " ".join(f"{s['kalinlik']}:{s['maske_iou']}" for s in satirlar if s["ad"] == o.ad),
            flush=True,
        )

    ad = f"duvar_deney_{a.bolum}{'_' + a.etiket if a.etiket else ''}.csv"
    with (SONUC / ad).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)

    print(f"\n{len(secilen)} plan, {time.perf_counter() - t_basla:.0f} s -> results/{ad}")
    gruplar = defaultdict(list)
    for s in satirlar:
        gruplar[(s["grup"], s["kalinlik"])].append(s)
    alanlar = ["maske_iou"] + ([] if a.yalniz_maske else ["parca_iou", "parca"]) + ["sure_s"]
    print("grup   kalınlık   n  " + "  ".join(f"{x:>24}" for x in alanlar))
    for (gr, k), ss in sorted(gruplar.items()):
        hucre = []
        for x in alanlar:
            v = np.array([s[x] for s in ss], float)
            hucre.append(
                f"{np.median(v):.3f} [{np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f}]"
            )
        print(f"{gr:6} {k:8} {len(ss):3}  " + "  ".join(f"{h:>24}" for h in hucre))


if __name__ == "__main__":
    main()
