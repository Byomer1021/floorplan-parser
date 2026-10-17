"""İnce duvar adayları ve özellikleri (K-007 için veri incelemesi).

Kalın maskeden (k = --kalin) arta kalan ince maske (k = --ince) Hough
parçalarına bölünür (``duvar.ince_aday_maskesi``); her parça için
``duvar.ince_ozellikler`` (uzunluk, kalınlık, kalınlık değişimi, doluluk,
paralel komşu) ve incelemeye ek olarak:

- uc_kalin_1/2: iki ucun en yakın kalın duvar pikseline uzaklığı (küçük önce)
- duvar_payi: parçanın dikdörtgeninin cevaptaki duvarla örtüşen payı (etiket)

Yalnız geliştirme, yalnız dolu duvarlı I-II planları (cevaptaki duvarın
Otsu dolgusu ≥ 0.8; stil seçimi değerlendirme içindir, sistem stili bilmez).

Kullanım: uv run python scripts/ince_aday_ozellik.py --ince 7
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.degerlendirme import parca_maskesi
from floorplan_parser.duvar import duvarlari_bul, ince_aday_maskesi, ince_ozellikler
from floorplan_parser.onisleme import delik_doldur, ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kalin", type=int, default=13)
    ap.add_argument("--ince", type=int, default=7)
    a = ap.parse_args()

    satirlar = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        cevap = maske(c, "Wall") > 0
        ikili = ikile(g)
        if (ikili[cevap] > 0).mean() < 0.8:
            continue
        kalin, aday = ince_aday_maskesi(delik_doldur(ikili, 4), a.kalin, a.ince)
        parcalar = duvarlari_bul(aday)
        kalin_uzaklik = cv2.distanceTransform(
            (kalin == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        h, w = g.shape
        for p in parcalar:
            oz = ince_ozellikler(p, aday, parcalar, 60)
            uc = sorted(
                float(kalin_uzaklik[min(max(round(y), 0), h - 1), min(max(round(x), 0), w - 1)])
                for x, y in (p.kapsam[:2], p.kapsam[2:])
            )
            pm = parca_maskesi([p], g.shape) > 0
            satirlar.append(
                {
                    "ad": o.ad,
                    **{k: round(v, 3) for k, v in oz.items()},
                    "uc_kalin_1": round(uc[0], 1),
                    "uc_kalin_2": round(uc[1], 1),
                    "alan": int(pm.sum()),
                    "duvar_payi": round(float(cevap[pm].mean()) if pm.any() else 0.0, 3),
                }
            )
        print(f"{o.ad:24} {len(parcalar)} ince parça", flush=True)

    ad = SONUC / f"ince_aday_ozellik_k{a.ince}.csv"
    with ad.open("w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        yazici.writeheader()
        yazici.writerows(satirlar)
    print(f"{len(satirlar)} parça -> results/{ad.name}")


if __name__ == "__main__":
    main()
