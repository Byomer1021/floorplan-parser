"""CVC-FP envanteri: plan başına boyut, nesne sayıları, duvar kalınlığı, duvar dolgusu.

Çıktı: results/cvcfp_envanter.csv ve gruplara göre özet (stdout).

- duvar_kalinlik: cevaptaki her duvar poligonu için 2·alan/çevre (uzun ince
  dikdörtgende kalınlığa eşit), plandaki medyan. Piksel.
- duvar_dolgu: cevaptaki duvar piksellerinin Otsu ile mürekkep çıkan payı.
  Dolu siyah duvar ~1, taralı orta, çift çizgili (içi boş) düşük.

Kullanım: uv run python scripts/cvcfp_envanter.py
"""

import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.onisleme import ikile

CIKTI = Path(__file__).resolve().parents[1] / "results" / "cvcfp_envanter.csv"


def kalinlik(noktalar: np.ndarray) -> float:
    p = noktalar.astype(np.float32)
    cevre = cv2.arcLength(p, closed=True)
    return 2 * cv2.contourArea(p) / cevre if cevre > 0 else 0.0


def main() -> None:
    t0 = time.perf_counter()
    satirlar = []
    iliski_tipleri: Counter = Counter()
    for o in ornekler():
        g, c = yukle(o)
        duvar = maske(c, "Wall") > 0
        murekkep = ikile(g) > 0
        iliski_tipleri.update(t for t, _ in c.iliskiler)
        satirlar.append(
            {
                "ad": o.ad,
                "grup": o.grup,
                "genislik": c.genislik,
                "yukseklik": c.yukseklik,
                "duvar": len(c.sinif("Wall")),
                "kapi": len(c.sinif("Door")),
                "pencere": len(c.sinif("Window")),
                "oda": len(c.sinif("Room")),
                "iliski": len(c.iliskiler),
                "duvar_kalinlik": round(
                    float(np.median([kalinlik(n.noktalar) for n in c.sinif("Wall")])), 1
                ),
                "duvar_dolgu": round(float(murekkep[duvar].mean()), 3),
                "duvar_alan_payi": round(float(duvar.mean()), 4),
            }
        )
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with CIKTI.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)

    print(
        f"{len(satirlar)} plan, {time.perf_counter() - t0:.0f} s, ilişki tipleri {dict(iliski_tipleri)}"
    )
    gruplar = defaultdict(list)
    for s in satirlar:
        gruplar[s["grup"]].append(s)
    alanlar = ["genislik", "duvar", "kapi", "pencere", "oda", "duvar_kalinlik", "duvar_dolgu"]
    print("grup     n  " + "  ".join(f"{a:>22}" for a in alanlar))
    for g, ss in gruplar.items():
        hucre = []
        for a in alanlar:
            v = np.array([s[a] for s in ss], float)
            hucre.append(f"{np.median(v):8.3g} [{v.min():.3g}-{v.max():.3g}]")
        print(f"{g:6} {len(ss):3}  " + "  ".join(f"{h:>22}" for h in hucre))
    toplam = {a: sum(s[a] for s in satirlar) for a in ("duvar", "kapi", "pencere", "oda")}
    print("toplam", toplam)


if __name__ == "__main__":
    main()
