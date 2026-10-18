"""Cevaptaki duvar poligonlarının ne kadarı bizim duvar maskemizde, açı ve
kalınlığa göre (geliştirme, I-II dolu duvarlı planlar).

Her cevap duvarı için en küçük döndürülmüş dikdörtgenden açı (yatay/dikeye
en yakın sapma, 0-45°) ve kalınlık (kısa kenar); kapsama = poligon
piksellerinin bizim maskede olan payı. Ayrıca cevap duvarının ham mürekkep
dolgusu (çizimde gerçekten dolu mu).

Çıktı: results/duvar_hata.csv; özet açı × kalınlık dilimlerinde.
Kullanım: uv run python scripts/duvar_hata.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from floorplan_parser.akis import plani_coz
from floorplan_parser.cvcfp import maske, ornekler, yukle
from floorplan_parser.onisleme import ikile

SONUC = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    satirlar = []
    for o in ornekler(bolum="gelistirme"):
        if o.grup != "I-II":
            continue
        g, c = yukle(o)
        ham = ikile(g)
        if (ham[maske(c, "Wall") > 0] > 0).mean() < 0.8:
            continue
        duvar = plani_coz(g).duvar_maske
        for n in c.sinif("Wall"):
            m = np.zeros(g.shape, np.uint8)
            cv2.fillPoly(m, [np.round(n.noktalar).astype(np.int32)], 1)
            if not m.any():
                continue
            (_, _), (w, h), aci = cv2.minAreaRect(n.noktalar.astype(np.float32))
            sapma = abs(aci) % 90
            sapma = min(sapma, 90 - sapma)
            satirlar.append(
                {
                    "ad": o.ad,
                    "sapma": round(sapma, 1),
                    "kalinlik": round(min(w, h), 1),
                    "uzunluk": round(max(w, h), 1),
                    "alan": int(m.sum()),
                    "kapsama": round(float((duvar[m > 0] > 0).mean()), 3),
                    "dolgu": round(float((ham[m > 0] > 0).mean()), 3),
                }
            )
        print(o.ad, flush=True)

    with (SONUC / "duvar_hata.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0]))
        w.writeheader()
        w.writerows(satirlar)

    def ozet(etiket, ss):
        if not ss:
            return
        alan = sum(s["alan"] for s in ss)
        kap = sum(s["alan"] * s["kapsama"] for s in ss) / max(alan, 1)
        kayip = sum(1 for s in ss if s["kapsama"] < 0.5)
        print(f"{etiket:28} n={len(ss):5} alan payı kapsama {kap:.3f}  kapsama<0.5: {kayip}")

    for aci_ad, alt, ust in (("eksen (<5°)", 0, 5), ("eğik (5-45°)", 5, 46)):
        for k_ad, k0, k1 in (
            ("<7", 0, 7),
            ("7-10", 7, 10),
            ("10-14", 10, 14),
            ("14-20", 14, 20),
            (">=20", 20, 1e9),
        ):
            ozet(
                f"{aci_ad} kalınlık {k_ad}",
                [s for s in satirlar if alt <= s["sapma"] < ust and k0 <= s["kalinlik"] < k1],
            )


if __name__ == "__main__":
    main()
